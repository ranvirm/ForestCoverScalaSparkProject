import DataCleaner.CleanData
import org.apache.spark.sql.DataFrame
// Do your data wrangling here - transformations, aggregations, joins etc.


object DataWrangler {

  def WrangledData(): DataFrame = {

    CleanData()

  }

}
