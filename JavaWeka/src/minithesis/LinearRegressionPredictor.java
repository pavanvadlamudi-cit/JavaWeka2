package minithesis;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.math.BigDecimal;

import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegressionPredictor {
	public static void main(String[] args) throws Exception {

		String datasetpath = "C:/CIT/Mini-Thesis/temperatureOnly/Processed/";
		String arffFolderName = "arff";
		String csvFolderName = "csv";
		String arffFileName =null;
		String csvFileName=null;
		String trainPath;
		String predictedOutFileFolderName = "csv";
		String predictedOutCSVFilename = null;
		File folder = new File(datasetpath + arffFolderName);
		File[] listOfFiles = folder.listFiles();
		String currentAttributeName=null;
		String outputPath = null;
		for (int i = 0; i < listOfFiles.length; i++) {
			currentAttributeName=null;
			outputPath = null;
			//if (i>=1) break;
			File file = listOfFiles[i];
			if (file.isFile() && file.getName().endsWith(".arff")) {
				System.out.println(file.getName().replace("arff", "csv"));
				// todo loop through each file
				arffFileName = file.getName();
				csvFileName = file.getName().replace("arff", "csv");
				trainPath = datasetpath + arffFolderName + "/"
						+ arffFileName;

				// read the data
				Instances data = new Instances(new BufferedReader(
						new FileReader(trainPath)));

				// Create LinearRegression object predictor
				LinearRegression predictor = new LinearRegression();

				if (data.numInstances() == 0) {
					System.out.println("Error: No instances found");
					return; // no data found in the file
				}
				System.out.printf("Number of attributes:%d\n",
						data.numAttributes());
				if (data.classIndex() == -1) {
					data.setClassIndex(data.numAttributes() - 1); // the
																	// target
																	// attribute
				}

				predictor.buildClassifier(data);

				if (data.classIndex() == -1) {
					data.setClassIndex(data.numAttributes() - 1);
				}

				int n = data.numInstances(), m = data.numAttributes();
				
				for(int attributeIndex=1;attributeIndex<m;attributeIndex++){
					if (data.attribute(attributeIndex).name()!=null)
						currentAttributeName = data.attribute(attributeIndex).name().replace("Hourly", "").replace("pro", "");
					else
						currentAttributeName ="";
					System.out.println(data.attribute(attributeIndex).name());
				
				double rmsle = 0;
				double rmsleZero = 0;

				outputPath= datasetpath + csvFolderName + "/"
						+ currentAttributeName+ csvFileName;

				try {
					// Create file
					FileWriter fstream = new FileWriter(outputPath);
					BufferedWriter out = new BufferedWriter(fstream);

					//int attributeindextoPredict = 1; // Last station
					// int attributeindextoPredict =2; //second station
					// int attributeindextoPredict =3; //first station
					double difference = 0;

					double rmslepred = 0;
					double rmsleact = 0;
					out.write("Instanceid,actual,predicted,difference");
					out.newLine();
					for (int curInstance = 0; curInstance < n; ++curInstance) {
						Instance t = data.instance(curInstance);
						double pred = predictor.classifyInstance(t), act = t.value(m - attributeIndex);
						BigDecimal pred1=new BigDecimal(String.valueOf(pred)).setScale(2, BigDecimal.ROUND_HALF_UP);
						//difference = act - pred;
						difference=act - pred1.doubleValue();
						BigDecimal difference1=new BigDecimal(String.valueOf(difference)).setScale(2, BigDecimal.ROUND_HALF_UP);
						
						rmslepred = (pred < 0 ? 0 : pred);
						rmsleact = (act < 0 ? 0 : act);
						// rmsle += (Math.log(pred + 1) - Math.log(act + 1)) *
						// (Math.log(pred + 1) - Math.log(act + 1));
						// The above formula gives error if pred variables less
						// <0 so modify pred to be zero on this instances
						rmsle += (Math.log(rmslepred + 1) - Math
								.log(rmsleact + 1))
								* (Math.log(rmslepred + 1) - Math
										.log(rmsleact + 1));
						rmsleZero += Math.log(rmsleact + 1)
								* Math.log(rmsleact + 1);
						// difference = act - pred;
						out.write((curInstance + 1) + "," + act + "," + pred1.doubleValue() + ","
								+ difference1);
						out.newLine();

						// System.out.println(act+":"+pred+":"+rmsle);

					}
					out.close();
					rmsle = Math.sqrt(rmsle / n);
					rmsleZero = Math.sqrt(rmsleZero / n);
					System.out.println("Successfully written prediction to a file  "
							+  outputPath);
					System.out.println("# of training data: "
							+ data.numInstances());
					System.out.println("RMSLE on testing data: " + rmsle);
					System.out.println("RMSLE on testing data Zero: "
							+ rmsleZero);
				} catch (Exception e) {// Catch exception if any
					System.err.println("Error: " + e.getMessage());
				}
				}
			}
		}
		// System.out.println("RMSLE on testing data: " + rmsle);
		// System.out.println("RMSLE on testing data Zero: " + rmsleZero);

	}
}
