
# 详细推导张正友标定法  
详细推导方法见代码中的paper，原文后面有我详细的一些注释，代码里面也是分模块写的，希望对大家的学习有帮助

# 结果对比
| 参数       | zhang（我们）                                                       | opencv                                                |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 相机内参   | fx=589.9421322083375<br />fy=589.0314402379594<br />u0=323.830691209651<br />v0=240.716451101392 | fx=589.9534727708858<br />fy=589.043571094065<br />u0=323.8345405252027<br />v0=240.7136136327917 |
| 畸变系数   | k1=0.08908862963423921 <br />k2=-0.0927595079832754                            | k1=0.08907964252174579<br />k2=-0.09266222570771097                            |
| 重投影误差 | 0.275026                                           | 0.274992                                          |

可以发现内外参和重投影误差的差距都在0.01以内，差距很小

# Reference

> 论文： "A Flexible New Technique for Camera Calibration" (2000).
>
> 参考：https://github.com/zhiyuanyou/Calibration-ZhangZhengyou-Method