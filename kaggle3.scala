// Author: Baichuan Zhang
// Code was written during the 2016 summer intern at Spark team in Hortonworks

import org.apache.spark.ml.feature.{PCA, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.types._
import scala.collection.mutable
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.clustering.KMeans
import scala.collection.mutable.ListBuffer
import scala.math.{sqrt => mysqrt}


// feature engineering works:
// 1) transform amount of money feature into log space: var38 log space
// 2) for each row, compute number of zero entries as a new feature
// 3) for feature var3, replace -999999 with its majority value which is 2
// 4) feature selection (remove zero variance features and remove duplicate features)
// 5) feature scaling --- z-normalization with dense vector
// 6) feature extraction with dimensionality reduction --- PCA
// 7) Use K-means to generate meta-features for classification task and you can refer to the paper for more details: "An Analysis of Single-Layer Networks in Unsupervised Feature Learning"
// 8) After meta-feature generation, do feature scaling again --- z-normalization with dense vector

// Under-sampling 20 times with 6 * labelRatio
// use random forest based classifier with default parameter setting

// Result: Public AUC = 0.797848; Private AUC = 0.781295

// --------------------------------------------------------------------

// Tuning parameters in the ML pipeline:
// 1) reduced dimensionality in PCA step
// 2) labelRatio during the under-sampling procedure
// 3) parameters in random forest based classifier

object kaggle3 {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("Kaggle Competition").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    // data loading from csv file
    val trainingData = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("/Users/baichuan.zhang/project/AD/data/kaggle/train.csv")

    // get the ratio for the pos / neg in the training set for further under-sampling purpose
    val posSampleCount = trainingData.filter(trainingData("TARGET") === 1).count()
    val negSampleCount = trainingData.count() - posSampleCount
    val labelRatio = 1.0 * posSampleCount / negSampleCount

    val testData = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("/Users/baichuan.zhang/project/AD/data/kaggle/test.csv")

    // transform the amount of money feature into log domain
    val allDataLog = trainingData.drop("TARGET").unionAll(testData).withColumn("new_var38", log(col("var38"))).drop("var38")

    // replace all -999999 with 2 in var3 feature column
    val replaceUDF = udf { input: Double => if (input == -999999) 2 else input }
    val allDataReplace = allDataLog.withColumn("var3Replace", replaceUDF(col("var3"))).drop("var3")

    val columnNames1 = allDataReplace.columns
    val assembler1 = new VectorAssembler().setInputCols(columnNames1.slice(1, columnNames1.length)).setOutputCol("features1")
    val df1 = assembler1.transform(allDataReplace)

    // for each row, compute the number of zero entries as a new feature to append to the existing data frame
    val nzUDF = udf { features: Vector => features.size - features.numNonzeros }
    val allData = df1.withColumn("nzFeaturesCount", nzUDF(col("features1")))

    val trainedFeature = Array("features1", "nzFeaturesCount")
    val assembler2 = new VectorAssembler().setInputCols(trainedFeature).setOutputCol("trainedFeatures")
    val df2 = assembler2.transform(allData).drop("features1")

    val columnNames2 = df2.columns

    val allDataRDD = df2.select(assembler2.getOutputCol).rdd.map { case Row(v: Vector) => v }
    val featureSummary = allDataRDD.aggregate(new MultivariateOnlineSummarizer())(
      (summary, feature) => summary.add(feature),
      (summary1, summary2) => summary1.merge(summary2)
    )

    val variances = featureSummary.variance

    // remove both features with zero variance and duplicated features
    val dict = mutable.Map.empty[Double, Int]
    val indices = scala.collection.mutable.MutableList[Int]()
    for (i <- 0 until variances.size) {
      if (variances(i) != 0.0 && !dict.contains(variances(i))) {
        dict(variances(i)) = i
        indices += i
      }
    }

    val arrayBuilder = mutable.ArrayBuilder.make[String]
    indices.foreach { i => arrayBuilder += columnNames2(i + 1) }
    val filteredColumnNames = arrayBuilder.result()

    // project the selected features into data frame
    val assembler3 = new VectorAssembler().setInputCols(filteredColumnNames).setOutputCol("features2")
    val df3 = assembler3.transform(df2)

    // convert df3 to dense matrix
    val toDenseUDF = udf { v: Vector => v.toDense }
    val df3Dense = df3.withColumn("features2Dense", toDenseUDF(col("features2"))).drop("features2")

    val scaler = new StandardScaler()
      .setInputCol("features2Dense")
      .setOutputCol("featureNorm")
      .setWithStd(true)
      .setWithMean(true)

    val scalerModel = scaler.fit(df3Dense)
    val df4 = scalerModel.transform(df3Dense)

    // perform PCA on scaled features
    val pca = new PCA()
      .setInputCol(scaler.getOutputCol)
      .setOutputCol("pcaFeatures")
      .setK(8)

    val pcaModel = pca.fit(df4)
    val df5 = pcaModel.transform(df4)

    // run K-means on the training set in order to generate meta-features for further classification task
    val kmeans = new KMeans()
      .setK(300)
      .setFeaturesCol(pca.getOutputCol)
      .setPredictionCol("prediction")

    val kmeansModel = kmeans.fit(df5)

    val centroids = kmeansModel.clusterCenters

    // use K-means clustering to generate features for classification task -- K-means soft triangle
    // paper resource: http://www.jmlr.org/proceedings/papers/v15/coates11a/coates11a.pdf

    import sqlContext.implicits._

    val kmeansDF = df5.select("ID", pca.getOutputCol).rdd.map { r =>
      val id = r.getInt(0)
      val dataPoint = r.getAs[Vector](1).toArray
      val distList = new ListBuffer[Double]
      (0 until centroids.size).foreach { i =>
        val dist = distance(dataPoint, centroids(i).toArray)
        distList += dist
      }
      (id, distList.toList)
    }.map { t =>
      val averageDist = t._2.sum / t._2.length
      val triangleList = new ListBuffer[Double]
      (0 until t._2.length).foreach { k =>
        val triangleValue = if (averageDist > t._2(k)) averageDist - t._2(k) else 0.0
        triangleList += triangleValue
      }
      (t._1, Vectors.dense(triangleList.toArray))
    }.toDF("ID", "GeneratedFeatures")

    // perform z-normalization on the generated meta features from K-means clustering
    val scaler2 = new StandardScaler()
      .setInputCol("GeneratedFeatures")
      .setOutputCol("GeneratedFeaturesNorm")
      .setWithStd(true)
      .setWithMean(true)

    val scalerModel2 = scaler2.fit(kmeansDF)
    val df6 = scalerModel2.transform(kmeansDF)

    val training = df6.join(trainingData, trainingData("ID") === df6("ID"))
      .select(col("GeneratedFeaturesNorm"), col("TARGET").cast(DoubleType))
    val test = df6.join(testData, testData("ID") === df6("ID"))
      .select(testData("ID"), col("GeneratedFeaturesNorm"))

    // because of extreme imbalanced positive and negative instances, perform under-sampling on training set in order to balance positive and negative instances
    val trainSamplePos = training.filter(training("TARGET") === 1.0)
    val trainSampleNeg = training.filter(training("TARGET") === 0.0).sample(true, 5 * labelRatio)
    val trainSample = trainSamplePos.unionAll(trainSampleNeg)

    // use the generated features for classification task
    // use random forest for classification

    val trainSample2 = new StringIndexer()
        .setInputCol("TARGET")
        .setOutputCol("indexedTarget")
        .fit(trainSample)
        .transform(trainSample)

    val rf = new RandomForestClassifier()
        .setLabelCol("indexedTarget")
        .setFeaturesCol(scaler2.getOutputCol)
        .setNumTrees(10)

    val rfModel = rf.fit(trainSample2)

    // save the test point ID and probability belonging to un-satisfactory customer into a file
    val modelPrediction = rfModel.transform(test).select(col("ID").cast(IntegerType), col("probability")).orderBy(col("ID")).rdd.map {
      case Row(id: Int, probability: Vector) =>
          (id, probability(1))
      }.toDF("ID", "probability2")
    modelPrediction.repartition(1).write.format("csv").save("/Users/baichuan.zhang/project/AD/kaggle_result/unsupervised_feature_learning/kmeans_dt_result")

    sc.stop()
  }

  // Compute the Euclidean distance between two vector points
  def distance(a: Array[Double], b: Array[Double]): Double = {
    mysqrt(a.zip(b).map(p => p._1 - p._2).map(d => d * d).sum)
  }
}
