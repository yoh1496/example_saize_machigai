using System;
using OpenCvSharp;

namespace dotnetcore_opencvsharp {
    class Program {
        static void Main(string[] args) {
            using var src = new Mat("example.jpg", ImreadModes.Grayscale);
            using var dst = new Mat();

            Cv2.Canny(src, dst, 50, 200);
            Cv2.ImWrite("example_canny.jpg", dst);
            Console.WriteLine("Example done!");
        }
    }
}