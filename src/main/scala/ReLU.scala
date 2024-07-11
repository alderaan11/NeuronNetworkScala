package org.jetbrains.scala

import botkop.numsca.{Tensor, maximum, zeros}


class ReLU(layer:  Layer_Dense){

  private var forward: Tensor = zeros(1, layer.getForward.shape.head)

  def setForward : Unit = {
    forward = maximum(layer.getForward, 0.00)
  }

  def getForward : Tensor = forward
  override def toString: String = {
    s"ReLU(output_result: $forward, forward: $forward)"
  }
}
