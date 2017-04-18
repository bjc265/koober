package edu.cs5152.predictionio.demandforecasting

import grizzled.slf4j.Logger
import org.apache.predictionio.controller.{CustomQuerySerializer, P2LAlgorithm, Params}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.joda.time.DateTime

case class AlgorithmParams(
  iterations:        Int    = 20,
  regParam:          Double = 0.1,
  miniBatchFraction: Double = 1.0, 
  stepSize:          Double = 0.001
) extends Params

class Algorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] with MyQuerySerializer {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, preparedData: PreparedData): Model = {
    preparedData.data.collect().foreach(println)

    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 3 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "variance"
    val maxDepth = 10
    val maxBins = 32

    val randomForestModel = RandomForest.trainRegressor(preparedData.data, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    new Model(randomForestModel, Preparator.locationClusterModel.get)
  }

  def predict(model: Model, query: Query): PredictedResult = {
    val label : Double = model.predict(query)
    new PredictedResult(label)
  }
}

class Model(mod: RandomForestModel, locationClusterModel: KMeansModel) extends Serializable { // will not be DateTime after changes
                                                                                  // to Preparator
  @transient lazy val logger = Logger[this.type]

  def predict(query: Query): Double = {
    val locationClusterLabel = locationClusterModel.predict(Vectors.dense(query.lat, query.lng))
    val features = Preparator.toFeaturesVector(DateTime.parse(query.eventTime), query.lat, query.lng, locationClusterLabel)
    mod.predict(features)
  }
}

trait MyQuerySerializer extends CustomQuerySerializer {
  @transient override lazy val querySerializer = org.json4s.DefaultFormats ++ org.json4s.ext.JodaTimeSerializers.all
}

