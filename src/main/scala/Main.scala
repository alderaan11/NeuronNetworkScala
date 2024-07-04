package org.jetbrains.scala
import botkop.{numsca => ns}
import ns.Tensor

object Main extends App{
  val inputs = Tensor(1,2,3,2.5) //The result is :  [1.00,  2.00,  3.00]
  val weights = Tensor(0.2,0.8,-0.5,1,0.5,-0.91,0.26,-0.5,-0.26,-0.27,0.17,0.87).reshape(3,4)


  val biais = Tensor(2,3,0.5)

  val output = ns.dot(inputs, weights.T)+biais

  println(s"The result is :  $inputs\n")
  println(s"The weights : $weights\n")
  println(s"Dot product : $output\n")





}

