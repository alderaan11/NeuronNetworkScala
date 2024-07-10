package org.jetbrains.scala
import botkop.{numsca => ns}
import ns.{Tensor, softmax}

///Tensor : object that can be represented as an array
//Array : ordered homologous container for numbers
//Vector : set of numbers in brackets


object Main extends App{
//  val inputs = Tensor(1,2,3,2.5,2, 5, -1, 2, -1.5, 2.7,3.3,-0.8).reshape(3,4) //The result is :  [1.00,  2.00,  3.00]
//  val weights = Tensor(0.2,0.8,-0.5,1,0.5,-0.91,0.26,-0.5,-0.26,-0.27,0.17,0.87).reshape(3,4)
//
//  val weights2 = Tensor(0.1,-0.14,0.5,-0.5,0.12,-0.33,-0.44,0.73,-0.13).reshape(3,3)
//    //line -> neuron, columns ->inputs
//
//
//  val biais = Tensor(2,3,0.5)
//  val biais2 = Tensor(-1,2,-0.5)
//
//  val output1 = ns.dot(inputs, weights.T)+biais
//  val output2 = ns.dot(output1, weights2.T)+biais2
//
//  println(s"The result is : $inputs\n")
//  println(s"The weights : $weights\n")
//  println(s"Dot product : $output1\n")
//  println(s"Dot product2 : $output2\n")

  val ld = new Layer_Dense(4, 3) // 4 inputs pour 3 neurones
  val inputs = Tensor(1, 2, 3, 2.5, 2, 5, -1, 2, -1.5, 2.7, 3.3, -0.8).reshape(3,4) //weights par neurones
  ld.setForward(inputs)
  println("Weight :")

  println(ld.getWeights)
  println("\nBiaises :")
  println(ld.getBiaises)
  println("\nOutput :")
  println(ld.getForward)

  val relu = new ReLU(ld)
  println(relu.getForward)

  val softMax = new SoftMax(ld)
  println(softMax.getForward)


}

