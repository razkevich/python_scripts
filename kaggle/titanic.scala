import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import java.io.FileWriter
import au.com.bytecode.opencsv.CSVParser
val dataWithHeader=sc.textFile("/input/titanic/train.csv")
val f=dataWithHeader.first
val data=dataWithHeader.filter(_!=f)

def parseLine(s:String)(noClass:Boolean)={
import au.com.bytecode.opencsv.CSVParser
val splitline = new CSVParser(',').parseLine(s);
LabeledPoint(
if (noClass) -1d else splitline(1).toDouble,
Vectors.dense(
splitline(0).toDouble,//id
if (splitline(2+(if (noClass) -1 else 0))=="1") 1d else 0d, //1 class
if (splitline(2+(if (noClass) -1 else 0))=="2") 1d else 0d, //2 class
if (splitline(2+(if (noClass) -1 else 0))=="3") 1d else 0d, //3 class
if (splitline(4+(if (noClass) -1 else 0))=="male") 1d else 0d, //male
if (splitline(4+(if (noClass) -1 else 0))=="female") 1d else 0d, //female
if (splitline(5+(if (noClass) -1 else 0))=="") -1d else splitline(5+(if (noClass) -1 else 0)).toDouble, //age
if ((if (splitline(5+(if (noClass) -1 else 0))=="") -1d else splitline(5+(if (noClass) -1 else 0)).toDouble)<5) 1 else 0, //age<5
if ((if (splitline(5+(if (noClass) -1 else 0))=="") -1d else splitline(5+(if (noClass) -1 else 0)).toDouble)<10) 1 else 0, //age<10
if ((if (splitline(5+(if (noClass) -1 else 0))=="") -1d else splitline(5+(if (noClass) -1 else 0)).toDouble)<15) 1 else 0, //age<15
if ((if (splitline(5+(if (noClass) -1 else 0))=="") -1d else splitline(5+(if (noClass) -1 else 0)).toDouble)<18) 1 else 0, //age<18
if ((if (splitline(5+(if (noClass) -1 else 0))=="") -1d else splitline(5+(if (noClass) -1 else 0)).toDouble)<23) 1 else 0, //age<23
if ((if (splitline(5+(if (noClass) -1 else 0))=="") -1d else splitline(5+(if (noClass) -1 else 0)).toDouble)<30) 1 else 0, //age<30
if ((if (splitline(5+(if (noClass) -1 else 0))=="") -1d else splitline(5+(if (noClass) -1 else 0)).toDouble)<40) 1 else 0, //age<40
if ((if (splitline(5+(if (noClass) -1 else 0))=="") -1d else splitline(5+(if (noClass) -1 else 0)).toDouble)<50) 1 else 0, //age<50
if ((if (splitline(5+(if (noClass) -1 else 0))=="") -1d else splitline(5+(if (noClass) -1 else 0)).toDouble)<60) 1 else 0, //age<60
if ((if (splitline(5+(if (noClass) -1 else 0))=="") -1d else splitline(5+(if (noClass) -1 else 0)).toDouble)<70) 1 else 0, //age<70
if ((if (splitline(5+(if (noClass) -1 else 0))=="") -1d else splitline(5+(if (noClass) -1 else 0)).toDouble)<80) 1 else 0, //age<80
if (splitline(6+(if (noClass) -1 else 0))=="") -1d else splitline(6+(if (noClass) -1 else 0)).toDouble, //sibsp
if (splitline(7+(if (noClass) -1 else 0))=="") -1d else splitline(7+(if (noClass) -1 else 0)).toDouble, //parch

(if (splitline(6+(if (noClass) -1 else 0))=="") -1d else splitline(6+(if (noClass) -1 else 0)).toDouble) + //sibsp
(if (splitline(7+(if (noClass) -1 else 0))=="") -1d else splitline(7+(if (noClass) -1 else 0)).toDouble),
if (splitline(8+(if (noClass) -1 else 0)).forall(Character.isDigit(_))) 1 else 0,
if (splitline(3+(if (noClass) -1 else 0)).find("Mrs")> -1) 1 else 0,
if (splitline(3+(if (noClass) -1 else 0)).find("Miss")> -1) 1 else 0,
if (splitline(9+(if (noClass) -1 else 0))=="") -1 else splitline(9+(if (noClass) -1 else 0)).toDouble
,if (splitline(10+(if (noClass) -1 else 0))=="") -1 else splitline(10+(if (noClass) -1 else 0))(0).toInt
,try{if (splitline(11+(if (noClass) -1 else 0))(0) == 'C') 1 else 0} catch{case _ => -1}
,try{if (splitline(11+(if (noClass) -1 else 0))(0) == 'S') 1 else 0} catch{case _ => -1}
,try{if (splitline(11+(if (noClass) -1 else 0))(0) == 'Q') 1 else 0} catch{case _ => -1}
)
)
}

val splits=data.map(x=>parseLine(x)(false)).randomSplit(Array(0.7,0.3),seed=11L)
val train=splits(0)
val test=splits(1)

val m=RandomForest.trainClassifier(train,2,Map[Int,Int](),200,"auto","entropy",30,50)
val predictionAndLabel=test.map(x=>(m.predict(x.features),x.label))
val accuracy=predictionAndLabel.filter(x=>x._1==x._2).count.toDouble/test.count.toDouble
val dataWithHeaderTest=sc.textFile("/input/titanic/test.csv")
val fTest=dataWithHeaderTest.first
val dataTest=dataWithHeaderTest.filter(_!=fTest) 

val splitsTest=dataTest.map(x=>parseLine(x)(true)).collect

val f=new FileWriter("/home/cloudera/out1.csv",true)
f.write(s"PassengerId,Survived\r\n")

splitsTest foreach (x=>{
val cl=m.predict(x.features).toString.replace(".0","")
val id=x.features(0).toString.replace(".0","")
f.write(s"${id},${cl}\r\n")
})
f.close
accuracy

