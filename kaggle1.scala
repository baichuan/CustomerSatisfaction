// Author: Baichuan Zhang
// Mentor: Yanbo Liang
// Code was written during the 2016 summer intern at Spark team in Hortonworks

import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.types._
import scala.collection.mutable
import org.apache.spark.ml.classification.DecisionTreeClassifier

// Model overview: We first perform various of feature engineering techniques.
// Considering the class imbalance issue in the dataset, we perform under-sampling for the dataset
// Use decision tree based classifier (Default Parameter Setting in Spark ML)
// Result: public AUC = 0.826238, private AUC = 0.811935

// --------------------------------------------------------------------

// Tuning parameters in the ML pipeline:
// 1) labelRatio during the under-sampling procedure
// 2) parameters in decision tree based classifier

object kaggle1 {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("Kaggle Competition").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    // load the data from a csv file
    val trainingData = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("/Users/baichuan.zhang/project/AD/data/kaggle/processed_train.csv")

    // compute the positive and negative data points ratio
    val posSampleCount = trainingData.filter(trainingData("TARGET") === 1).count()
    val negSampleCount = trainingData.count() - posSampleCount
    val labelRatio = 1.0 * posSampleCount / negSampleCount

    val testData = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("/Users/baichuan.zhang/project/AD/data/kaggle/processed_test.csv")

    val allDataTemp = trainingData.drop("TARGET").unionAll(testData)

    // transform the amount of money feature into log space
    val allData = allDataTemp.withColumn("new_var38", log(col("var38"))).drop("var38")

    val columnNames = allData.columns
    val assembler1 = new VectorAssembler().setInputCols(columnNames.slice(1, columnNames.length)).setOutputCol("features1")
    val df1 = assembler1.transform(allData)

    // for each row, compute the number of zero entries as a new feature to append to existing data frame
    val nzUDF = udf { features: Vector => features.size - features.numNonzeros }
    val allData2 = df1.withColumn("nzFeaturesCount", nzUDF(col("features1")))

    val allDataRDD = df1.select(assembler1.getOutputCol).rdd.map { case Row(v: Vector) => v }
    val featureSummary = allDataRDD.aggregate(new MultivariateOnlineSummarizer())(
      (summary, feature) => summary.add(feature),
      (summary1, summary2) => summary1.merge(summary2)
    )

    val variances = featureSummary.variance

    // remove features with zero variance and duplicated features
    val dict = mutable.Map.empty[Double, Int]
    val indices = scala.collection.mutable.MutableList[Int]()
    for (i <- 0 until variances.size) {
      if (variances(i) != 0.0 && ! dict.contains(variances(i))) {
        dict(variances(i)) = i
        indices += i
      }
    }

    val arrayBuilder = mutable.ArrayBuilder.make[String]
    indices.foreach { i => arrayBuilder += columnNames(i + 1) }
    val filteredColumnNames = arrayBuilder.result()

    // project the selected features into data frame
    val assembler2 = new VectorAssembler().setInputCols(filteredColumnNames).setOutputCol("features2")
    val df2 = assembler2.transform(allData2)

    val trainedFeature = Array("features2", "nzFeaturesCount")
    val assembler3 = new VectorAssembler().setInputCols(trainedFeature).setOutputCol("trainedFeatures")
    val df3 = assembler3.transform(df2)

    // perform feature scaling --- z-normalization
    val scaler = new StandardScaler()
      .setInputCol(assembler3.getOutputCol)
      .setOutputCol("featureNorm")
      .setWithStd(true)
      .setWithMean(false)

    val scalerModel = scaler.fit(df3)
    val df4 = scalerModel.transform(df3)

    val training = df4.join(trainingData, trainingData("ID") === allData("ID")).select(col(scaler.getOutputCol), col("TARGET").cast(DoubleType))
    val test = df4.join(testData, testData("ID") === allData("ID")).select(testData("ID"), col(scaler.getOutputCol))
    val trainSamplePos = training.filter(training("TARGET") === 1.0)

    // because of extreme imbalanced positive and negative instances, so do under-sampling on training dataset in order to balance positive and negative instances
    // run the decision tree classifier 20 times in order to capture entire information of data
    for (i <- 0 until 20) {

      val trainSampleNeg = training.filter(training("TARGET") === 0.0).sample(true, 5 * labelRatio)
      val trainSample = trainSamplePos.unionAll(trainSampleNeg)

      val trainSample2 = new StringIndexer()
        .setInputCol("TARGET")
        .setOutputCol("indexedTarget")
        .fit(trainSample)
        .transform(trainSample)

      val dt = new DecisionTreeClassifier()
        .setLabelCol("indexedTarget")
        .setFeaturesCol(scaler.getOutputCol)

      val dtModel = dt.fit(trainSample2)

      import sqlContext.implicits._

      val modelPrediction = dtModel.transform(test).select(col("ID").cast(IntegerType), col("probability")).orderBy(col("ID")).rdd.map {
        case Row(id: Int, probability: Vector) =>
          (id, probability(1))
      }.toDF("ID", "probability2")
      modelPrediction.repartition(1).write.format("csv").save("/Users/baichuan.zhang/project/AD/kaggleResult" + i)
    }
    sc.stop()
  }
}
