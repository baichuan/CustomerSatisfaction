// Author: Baichuan Zhang
// Code was written during the 2016 summer intern at Spark team in Hortonworks

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, StandardScaler}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.types._
import scala.collection.mutable
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer

// feature engineering works:
// 1) remove duplicate rows and resolve label inconsistency issue
// 2) transform amount of money feature into log space: var38 log space
// 3) for each row, compute number of zero entries as a new feature
// 4) for feature var3, replace -999999 with its majority value which is 2
// 5) feature selection (remove zero variance features and remove duplicate features)
// 6) feature scaling --- z-normalization

// Under-sampling 20 times with 6 * labelRatio
// use decision tree based classifier with default parameter setting

// Result: Public AUC = 0.824460; Private AUC = 0.809471

// --------------------------------------------------------------------

// Tuning parameters in the ML pipeline:
// 1) labelRatio during the under-sampling procedure
// 2) parameters in decision tree based classifier

object kaggle2 {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("Kaggle Competition").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    // load the data from csv file
    val trainDataOne = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("/Users/baichuan.zhang/project/AD/data/kaggle/train.csv")

    // remove duplicate rows and resolve label inconsistency issue
    val columnNamesOne = trainDataOne.columns
    val assemblerOne = new VectorAssembler()
      .setInputCols(columnNamesOne.slice(1, columnNamesOne.length -1))
      .setOutputCol("features1")
    val trainDataVA1 = assemblerOne.transform(trainDataOne)

    val groupbyTrainData = trainDataVA1
      .groupBy(col("features1"))
      .agg(sum(col("TARGET")), min(col("ID")))

    // resolve label inconsistency issue
    import sqlContext.implicits._
    val trainDataTwo = groupbyTrainData.select("min(ID)", "sum(TARGET)").rdd.map { r =>
      val minID = r.getInt(0)
      val label = r.getLong(1)
      val resolvedLabel = if (label >= 0.0) 1.0 else -1.0
      (minID, resolvedLabel)
    }.toDF("minID", "resolvedLabel")

    // join trainDataOne and trainDataTwo based on the id in order to remove duplicate rows
    val schemaNames = scala.collection.mutable.ArrayBuffer.empty[String]
    schemaNames += "minID"
    for (i <- 1 until columnNamesOne.size - 1) {
      schemaNames += columnNamesOne(i)
    }
    schemaNames += "resolvedLabel"
    val schemaNamesArray = schemaNames.toArray
    val trainingData = trainDataOne.join(trainDataTwo, trainDataOne("ID") === trainDataTwo("minID"))
      .select(schemaNamesArray.map(col(_)):_*)

    // get the ratio for the pos / neg in the training set for further under-sampling step
    val posSampleCount = trainingData.filter(trainingData("resolvedLabel") === 1.0).count()
    val negSampleCount = trainingData.count() - posSampleCount
    val labelRatio = 1.0 * posSampleCount / negSampleCount

    val testData = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("/Users/baichuan.zhang/project/AD/data/kaggle/test.csv")

    // transform the amount of money feature into log domain
    val allDataLog = trainingData.drop("resolvedLabel").unionAll(testData).withColumn("new_var38", log(col("var38"))).drop("var38")

    // replace all -999999 with 2 in var3 feature column
    val replaceUDF = udf { input: Double => if (input == -999999) 2 else input }
    val allDataReplace = allDataLog.withColumn("var3Replace", replaceUDF(col("var3"))).drop("var3")

    val columnNames1 = allDataReplace.columns
    val assembler1 = new VectorAssembler().setInputCols(columnNames1.slice(1, columnNames1.length)).setOutputCol("features2")
    val df1 = assembler1.transform(allDataReplace)

    // for each row, compute the number of zero entries as a new feature to append to the existing data frame
    val nzUDF = udf { features: Vector => features.size - features.numNonzeros }
    val allData = df1.withColumn("nzFeaturesCount", nzUDF(col("features1")))

    val trainedFeature = Array("features2", "nzFeaturesCount")
    val assembler2 = new VectorAssembler().setInputCols(trainedFeature).setOutputCol("trainedFeatures")
    val df2 = assembler2.transform(allData).drop("features2")

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
      if (variances(i) >= 0.0 && !dict.contains(variances(i))) {
        dict(variances(i)) = i
        indices += i
      }
    }

    val arrayBuilder = mutable.ArrayBuilder.make[String]
    indices.foreach { i => arrayBuilder += columnNames2(i + 1) }
    val filteredColumnNames = arrayBuilder.result()

    val assembler3 = new VectorAssembler().setInputCols(filteredColumnNames).setOutputCol("features3")
    val df3 = assembler3.transform(df2)

    // feature scaling step
    val scaler = new StandardScaler()
      .setInputCol(assembler3.getOutputCol)
      .setOutputCol("featureNorm")
      .setWithStd(true)
      .setWithMean(false)

    val scalerModel = scaler.fit(df3)
    val df4 = scalerModel.transform(df3)

    val training = df4.join(trainingData, trainingData("minID") === allData("minID")).select(col(scaler.getOutputCol), col("resolvedLabel").cast(DoubleType))

    // because of extreme imbalanced positive and negative instances, so do under-sampling on training dataset in order to balance positive and negative instances
    // run 20 times in order to capture all data information

    val trainSamplePos = training.filter(training("resolvedLabel") === 1.0)
    val test = df4.join(testData, testData("ID") === allData("minID")).select(testData("ID"), col(scaler.getOutputCol))

    for (i <- 0 until 20) {

      val trainSampleNeg = training.filter(training("resolvedLabel") === -1.0).sample(true, 6 * labelRatio)
      val trainSample = trainSamplePos.unionAll(trainSampleNeg)

      val trainSample2 = new StringIndexer()
        .setInputCol("resolvedLabel")
        .setOutputCol("indexedTarget")
        .fit(trainSample)
        .transform(trainSample)

      val dt = new DecisionTreeClassifier()
        .setLabelCol("indexedTarget")
        .setFeaturesCol(scaler.getOutputCol)

      val dtModel = dt.fit(trainSample2)

      val modelPrediction = dtModel.transform(test).select(col("ID").cast(IntegerType), col("probability")).orderBy(col("ID")).rdd.map {
        case Row(id: Int, probability: Vector) =>
          (id, probability(1))
      }.toDF("ID", "probability2")

      modelPrediction.repartition(1).write.format("csv").save("/Users/baichuan.zhang/project/AD/kaggleResult" + i)
    }
    sc.stop()
  }
}