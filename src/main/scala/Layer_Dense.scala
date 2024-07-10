package org.jetbrains.scala

import botkop.numsca.{rand, zeros}
import botkop.numsca.Tensor
import botkop.numsca._

class Layer_Dense(n_inputs : Int, n_neurons : Int){

  private val inputs: Int = n_inputs
  private val nneurons: Int = n_neurons
  private var weights: Tensor = 0.01 * randn(inputs, nneurons)
  private val biases: Tensor = zeros(1, n_neurons)
  private var forward: Tensor = zeros(1, n_neurons) // Initialiser output comme un Tensor

  // Getters
  def getWeights: Tensor = weights
  def getBiaises: Tensor = biases
  def getForward: Tensor = forward
  def getInputs: Int = inputs
  def getNneurons : Int = nneurons


  def setForward(inputs: Tensor): Unit = {
    forward = dot(inputs, weights) + biases
  }

  // toString method
  override def toString: String = {
    s"Layer_Dense(weights: ${weights}, biases: ${biases},  output of the layer: ${forward})"
  }

}
