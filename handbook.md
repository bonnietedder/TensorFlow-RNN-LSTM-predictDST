##		研究对象

采用由**NASA OMNIWeb**提供的太阳风参数及地磁活动指数的公开数据，对第23太阳活动周期(1996-01-01—2008-12-31)的数据包括**行星际磁场*Bz*分量、太阳风速度、太阳风质子密度**等重要太阳风参数及***D*st指数**等主要的地磁指数进行统计分析。

行星际磁场*Bz*分量(IMF *Bz)*、太阳风质子密度(*N*sw)、太阳风速度(*V*sw)。

采用数据的时间分辨率为1 hour。



##  研究数据A

研究数据A包括：**行星际磁场*Bz*分量(IMF *Bz)*、太阳风质子密度(*N*sw)、太阳风速度(*V*sw)**

来源：https://spdf.sci.gsfc.nasa.gov/pub/data/omni/low_res_omni/

OMNI2_YYYY.DAT其中YYYY是给定的一年。OMNIYYYY.DAT文件，包含IMF和太阳风速矢量的原始数据。GSM坐标系。

OMNI_MYYYY.DAT文件是通过重新格式化从这些文件中创建的。RTN坐标系。   

### 数据格式

```matlab
#OMNI2_YYYY.DAT FORMAT DESCRIPTION  
```

![format_A](/Users/apple/Documents/毕设/format_A.png)



#### 所需要的数据：

| WORD | FORMAT | Fill Value | MEANING             | UNITS/COMMENTS |
| ---- | ------ | ---------- | ------------------- | -------------- |
| 13   | F6 1   | 999.9      | Bx GSE,GSM          | nT             |
| 14   | F6.1   | 999.9      | By GSE              | nT             |
| 15   | F6.1   | 999.9      | Bz GSE              | nT             |
| 16   | F6.1   | 999.9      | By GSM              | nT             |
| 17   | F6 1   | 999.9      | Bz GSM              | nT             |
| 24   | F6.1   | 999.9      | Proton Density      | N/cm^3         |
| 25   | F6.0   | 9999.      | Plasma (Flow) speed | km/s           |
| 41   | I6     | 99999      | DST Index           | nT             |

A，B，C分别对应年，日，时；

Q，X，Y分别对应行星际磁场*Bz*分量(IMF *Bz)*、太阳风质子密度(*N*sw)、太阳风速度(*V*sw)；

AO对应Dst指数。

13 14 15 16对应m,n,o,p

### 数据密度

频率为每小时一次，每个dat代表一整年，即365*24



##		研究数据B

研究数据B包括：**Dst指数**

来源：http://wdc.kugi.kyoto-u.ac.jp/dst_final/index.html

 **Graduate School of Science, Kyoto University**

​                     

### 数据格式        

每个月份：31*24        

![format_Dst](/Users/apple/Documents/毕设/format_Dst.png)

取出一天一小时单独查看，共26个数据，可去头去尾读取：

```matlab
DST6301*01  
X219 
000
-006-005-005-003-003-006-008-009-006-002-002-005-006-008-006-011-017-016-012-013-012-015-017-012
-009
```

### 数据密度

365*24