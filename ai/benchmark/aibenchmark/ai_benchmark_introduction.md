
<fcolor="LightSeaGreen"> Intruduction </font>
----

AI Benchmark is focused on performance , not dig in accuracy

Hardware
----

- Qualcomm chipsets / SNPE SDK

- HiSilicon chipsets / HIAI SDK

- MediaTek chipsets / NeuroPilot SDK

- Arm Cortex CPUs / Mali GPUs / NN SDK

Deep Learning Test [version 2.0.0]
----

#### Test Details
- Test1: Image Recognition (MobileNet-V1)
- Test2: Image Recognition (Inception-V3)
- Test3: Face Recognition (Inception-Resnet-V1)
- Test4: Image Delurring (SRCNN)
- Test5: Image Super-Resolution (VGG-19)
- Test6: Image Super-Resolution (ResNet-16)
- Test7: Image Semantic Segmentation (ICNet)
- Test8: Image Enhancement (ResNet-4)
- Test9: Memory Limitations (SRCNN)

#### Technical Description
- Test 1,2,4,5,8,9 use NNAPI, whileTest 3,6,7 executed on CPU
- first 8 tests has a predefined time limit : 25, 40, 40, 30, 40, 50, 20, 25 seconds, computed as an average(when more than two images processed within limit time , the first two is not considered)
- The 9th test does not have a time limit -- images of increasing resolution are processed untill the device runs out of memory.
- The final socre is caclulated as a weighted sum of scores obtained in these 9 tests. The weigt coefficients were calibrated based on the results obtained on Google Pixel2 with CPU.

Results
----

#### Qualcomm (Snapdragon 845)
- Provide hardware acceleration for quantized models, float models are not yet support
- Float models may use Adreno (Aderno 506 130 GFLOPs  Aderno 530 727 GFLOPs)

#### Huawei (Kirin 970)
- Lack of acceleration support for quantized models.
- CPU performance is quite similar to other chipset with the same Cortex.

#### MediaTek (P60)
- NNAPI both enable for float and qantized models.

Descussion
----

- Quantized model: There is no standard and reliable tools for quantizing network models.




