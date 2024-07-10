package org.jetbrains.scala

import botkop.numsca.{Tensor, exp, max, maximum, prod, sum, zeros}

class SoftMax(layer : Layer_Dense) {

  private var forward: Tensor = zeros(1, layer.getForward.shape.head)

  def setForward: Unit = {
    val exp_values = exp(layer.getForward - max(layer.getForward, axis=1))
    forward = exp_values / sum(exp_values, axis=1)
  }

  def getForward: Tensor = forward

  override def toString: String = {
    s"ReLU(output_result: $forward, forward: $forward)"
  }
}
