using System;
using System.Linq;
using OpenCvSharp;

namespace dotnetcore_opencvsharp {
    class Program {
        static void Main(string[] args) {
            using var srcA = new Mat("./input/A.jpg", ImreadModes.Color);
            using var srcB = new Mat("./input/B.jpg", ImreadModes.Color);
            using var maskA = new Mat("./input/A_mask.jpg", ImreadModes.Grayscale);
            using var maskB = new Mat("./input/B_mask.jpg", ImreadModes.Grayscale);

            var sift = OpenCvSharp.Features2D.SIFT.Create();
            KeyPoint[] kp1, kp2;
            Mat des1 = new Mat(), des2 = new Mat();
            sift.DetectAndCompute(srcA, maskA, out kp1, des1);
            sift.DetectAndCompute(srcB, maskB, out kp2, des2);

            var bf = new BFMatcher();
            var matches = bf.KnnMatch(des1, des2, 2);

            // ratio test
            double ratio = 0.8;
            var good = matches.Where( match => {
                if (match[0].Distance < ratio * match[1].Distance) return true;
                return false;
            }).Select( match => new DMatch[1] {match[0]});

            Mat result = new Mat();
            Cv2.DrawMatchesKnn(srcA, kp1, srcB, kp2, good, result);
            
            Cv2.ImWrite("result.png", result);

            var ptsA = good.Select(m => kp1[m[0].QueryIdx].Pt).ToList();
            var ptsB = good.Select(m => kp2[m[0].TrainIdx].Pt).ToList();
            
            var H = Cv2.FindHomography(InputArray.Create(ptsA),InputArray.Create(ptsB), HomographyMethods.Ransac, 5.0);
            
            Mat warped = new Mat();
            Cv2.WarpPerspective(srcA, warped, H, srcA.Size() );
            Cv2.ImWrite("warped_0.png", warped);
            Cv2.ImWrite("warped_1.png", srcB);
            Console.WriteLine("Example done!");
        }
    }
}