package org.jetbrains.scala
import botkop.numsca.Tensor
import botkop.numsca._
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
object Spiral_Data {



    // Fonction sin pour un Tensor
    def sin(t: Tensor): Tensor = new Tensor(Transforms.sin(t.array))

    def cos(t: Tensor): Tensor = new Tensor(Transforms.cos(t.array))
    //def tan(t: Tensor): Tensor = new Tensor(Transforms.tan(t.array))

    def arcsin(t: Tensor): Tensor = new Tensor(Transforms.asin(t.array))

    def arccos(t: Tensor): Tensor = new Tensor(Transforms.acos(t.array))

    def arctan(t: Tensor): Tensor = new Tensor(Transforms.atan(t.array))



    def generate(samples: Int, classes: Int): (Tensor, Tensor) = {
        val X = zeros(samples * classes, 2)
        val y = zeros(samples * classes)

        for (classNumber <- 0 until classes) {
            val ix = (samples * classNumber) until (samples * (classNumber + 1))

            // Génération de r avec linspace et reshape
            val r = linspace(0.0, 1.0, samples).reshape(samples, 1)

            // Génération de t avec linspace et uniform
            val samples_array = Array.tabulate(samples)(i => i)
            val t = linspace(classNumber * 4.0, (classNumber + 1) * 4.0, samples)

            // Calcul de sin(t) et cos(t) et reshape
            val sin_t = sin(t).reshape(samples, 1)
            val cos_t = cos(t).reshape(samples, 1)

            for (i <- ix) {
                val j = i - samples * classNumber
                X(i, 0) := r(j, 0) * sin_t(j, 0)
                X(i, 1) := r(j, 0) * cos_t(j, 0)
                y(i) := classNumber.toDouble
            }
        }

        (X, y)
    }

}
