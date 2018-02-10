import FeatureEngineering.FeatureData
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassifier}
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame

// Do final data preparations for machine learning here
// Define and run machine learning models here
// This should generally return trained machine learning models and or labelled data

object MachineLearning {

  def MachineLearningOutput(): DataFrame = {

    // name 'survived' column as label column
    val inputData = FeatureData().withColumnRenamed("Cover_Type", "label")

    // define feature columns
    val featureCols = inputData.drop("label").columns

    // define feature vector assembler
    val featureAssembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val inputDataWithFeatureVector = featureAssembler.transform(inputData)

    // def label indexer
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabels")
      .fit(inputDataWithFeatureVector)

    // def features indexer
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .fit(inputDataWithFeatureVector)

    // split data into train and test
    val Array(trainData, testData) = inputDataWithFeatureVector.randomSplit(weights = Array[Double](0.7, 0.3))

    // Def RandomForest model
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabels")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(20)
      .setMaxDepth(30)
      .setImpurity("entropy")
      .setSeed(1L)

    // Convert indexed labels back to original labels
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Chain indexers and forest in a Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    // Train model. This also runs the indexers
    val model = pipeline.fit(trainData)

    // Make predictions
    val predictions = model.transform(testData)

    // feature importance
    val featureImportance = model
      .stages(2)
      .asInstanceOf[RandomForestClassificationModel]
      .featureImportances

    //println(featureImportance)

    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabels")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)

    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("Test Error = " + (1.0 - accuracy))
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")

    predictions

  }

}
