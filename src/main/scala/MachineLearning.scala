import FeatureEngineering.FeatureData
import DataWrangler.WrangledData
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{count, when}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

// Do final data preparations for machine learning here
// Define and run machine learning models here
// This should generally return trained machine learning models and or labelled data

object MachineLearning {

  def RandomForestModel(): PipelineModel = {

    // name 'cover type' column as label column
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

    //println(featureImportance)

    // print feature importance
    featureImportance.toArray.zipWithIndex
      .map(_.swap)
      //.sortBy(-_._2)
      .foreach(x => println(x._1 + " -> " + x._2))

    // print feature names
    featureCols.foreach(x => println(x))

    // define evaluator to calculate test error
    val evaluatorAccuracy = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabels")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    // define evaluator to calculate
    val evaluatorF1 = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabels")
      .setPredictionCol("prediction")
      .setMetricName("f1")

    // define evaluator to calculate
    val evaluatorWeightedPrecision = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabels")
      .setPredictionCol("prediction")
      .setMetricName("weightedPrecision")

    // define evaluator to calculate
    val evaluatorWeightedRecall = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabels")
      .setPredictionCol("prediction")
      .setMetricName("weightedRecall")

    // calculate test error
    val modelAccuracy = evaluatorAccuracy.evaluate(predictions)

    // calculate model f1
    val f1 = evaluatorF1.evaluate(predictions)

    // calculate weighted precision
    val weightedPrecision = evaluatorWeightedPrecision.evaluate(predictions)

    // calculate weighted recall
    val weightedRecall = evaluatorWeightedRecall.evaluate(predictions)

    // print evaluators
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("Test Error = " + (1.0 - modelAccuracy))
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("F1 = " + f1)
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    print("Weighted Precision = "+weightedPrecision)
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("Weighted Recall = "+weightedRecall)
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")
    println("+++++++++++++++++")

    // define function to calculate confusion matrix
    def confusionMatrix(dataFrame: DataFrame): Unit = {

      val inData = dataFrame
        .select("label", "predictedLabel")

      val outData = inData
        .withColumn("Outcome", when(inData("label")===inData("predictedLabel"), "Correct").otherwise("Incorrect"))

      println("++++ Grouped by Actual Label")
      outData
        .groupBy("label", "Outcome")
        .agg(count("Outcome"))
        .show(20)

      println("++++ Grouped by Predicted Label")
      outData
        .groupBy("predictedLabel", "Outcome")
        .agg(count("Outcome"))
        .show(20)

    }

    // print confusion matrix
    confusionMatrix(predictions)

    /**
    val metrics = new MulticlassMetrics(predictions
      .select("predictedLabel","label")
      .rdd.map(r=>(r.getAs[DenseVector](0)(1),r.getDouble(1))))

    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    // Overall Statistics
    val accuracy = metrics.accuracy
    println("Summary Statistics")
    println(s"Accuracy = $accuracy")

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }

    // Recall by label
    labels.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }

    // False positive rate by label
    labels.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }

    // F-measure by label
    labels.foreach { l =>
      println(s"F1-Score($l) = " + metrics.fMeasure(l))
    }

    // Weighted stats
    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}")
    println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")

    */

    // return labeled data frame
    //predictions

    // return model
    model

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
