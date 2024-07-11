package org.jetbrains.scala

import botkop.numsca.Tensor
import breeze.linalg.softmax
import Spiral_Data._

///Tensor : object that can be represented as an array
//Array : ordered homologous container for numbers
//Vector : set of numbers in brackets


object Main extends App{

  def printTensor5 (t: Tensor) = {
    for (i <- 0 until 5) {
      println(t(i))
    }
  }

  val (x, y) = Spiral_Data.generate(samples = 100, classes = 3)

  var dense1 : Layer_Dense = new Layer_Dense(2,3)
  dense1.setForward(x)

  val activation1 : ReLU = new ReLU(dense1)
  activation1.setForward


  var dense2 : Layer_Dense = new Layer_Dense(3,3)
  dense2.setForward(activation1.getForward)

  var activation2 : SoftMax = new SoftMax(dense2)
  activation2.setForward

  val output = activation2.getForward
  printTensor5(output)


}

