import FeatureEngineering.FeatureData
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame

// Do final data preparations for machine learning here
// Define and run machine learning models here
// This should generally return trained machine learning models and or labelled data

object MachineLearning {

  def RandomForestModel(): DataFrame = {

    // name 'survived' column as label column
    val inputData = FeatureData().withColumnRenamed("Cover_Type", "label")

    // define feature columns
    val featureCols = inputData.drop("label").columns

    // define feature vector assembler
    val featureAssembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val inputDataWithFeatureVector = featureAssembler.transform(inputData)

    // define label indexer
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabels")
      .fit(inputDataWithFeatureVector)

    // define features indexer
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .fit(inputDataWithFeatureVector)

    // split data into train and test
    val Array(trainData, testData) = inputDataWithFeatureVector.randomSplit(weights = Array[Double](0.7, 0.3))

    //  define randomForest model
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabels")
      .setFeaturesCol("indexedFeatures")
      .setSeed(1L)
      .setNumTrees(20)
      .setMaxDepth(30)
      .setImpurity("entropy")

    // define function to convert indexed labels back to original labels
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // define pipeline model
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    // train model
    val model = pipeline.fit(trainData)

    // use trained model to make predictions on test data
    val predictions = model.transform(testData)

    // feature importance
    val featureImportance = model
      .stages(2)
      .asInstanceOf[RandomForestClassificationModel]
      .featureImportances

    println(featureImportance)

    // define evaluator to calculate test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabels")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    // calculate test error
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

  def RandomForestCVModel(): DataFrame = {

    // name 'survived' column as label column
    val inputData = FeatureData().withColumnRenamed("Cover_Type", "label")

    // define feature columns
    val featureCols = inputData.drop("label").columns

    // define feature vector assembler
    val featureAssembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val inputDataWithFeatureVector = featureAssembler.transform(inputData)

    // define label indexer
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabels")
      .fit(inputDataWithFeatureVector)

    // define features indexer
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .fit(inputDataWithFeatureVector)

    // split data into train and test
    val Array(trainData, testData) = inputDataWithFeatureVector.randomSplit(weights = Array[Double](0.7, 0.3))

    //  define randomForest model
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabels")
      .setFeaturesCol("indexedFeatures")
      .setSeed(1L)
    //.setNumTrees(20)
    //.setMaxDepth(30)
    //.setImpurity("entropy")

    // define function to convert indexed labels back to original labels
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // define pipeline model
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    // define grid of parameters to search over
    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.numTrees, Array[Int](10, 20))
      .addGrid(rf.maxDepth, Array[Int](20, 25, 30))
      .addGrid(rf.impurity, Array[String]("gini", "entropy"))
      .build()

    // define cross-validation model to run pipeline with param-search
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator()
        .setLabelCol("indexedLabels")
        .setPredictionCol("prediction")
        .setMetricName("accuracy"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    // run cross-validation model and choose the best set of parameters for model
    val cvModel = cv.fit(trainData)

    // show optimal params

    // use trained cv model to make predictions on test data
    val predictions = cvModel.transform(testData)

    // train model
    //val model = pipeline.fit(trainData)

    // use trained model to make predictions on test data
    //val predictions = model.transform(testData)

    // feature importance
    //val featureImportance = model
    //  .stages(2)
    //  .asInstanceOf[RandomForestClassificationModel]
    //  .featureImportances

    //println(featureImportance)

    // define evaluator to calculate test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabels")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    // calculate test error
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
