import MachineLearning.RandomForestModel
// save your output from the various objects here

object OutputSaver {

  // def function to save random forest pipeline model
  def RandomForestPipeLineModelSave(): Unit = {

    // save random forest pipeline model
    RandomForestModel().write.overwrite().save("./src/main/output/randomForestPipelineModel")

  }

}
