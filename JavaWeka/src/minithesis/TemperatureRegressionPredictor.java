package minithesis;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.math.BigDecimal;

import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class TemperatureRegressionPredictor {
	static String arffFolderName = "arff";
	static String csvFolderName = "csv";
	static String arffFileName = null;
	static String csvFileName = null;
	static String trainPath;
	static String predictedOutFileFolderName = "csv";
	static String predictedOutCSVFilename = null;

	static String currentAttributeName = null;
	static String outputPath = null;

	
	public static void MultiLayerPerceptroPredictorClassifier(String datasetPath) {
		try {
			File folder = new File(datasetPath + arffFolderName);
			File[] listOfFiles = folder.listFiles();
			for (int i = 0; i < listOfFiles.length; i++) {
				currentAttributeName = null;
				outputPath = null;
				
				File file = listOfFiles[i];
				if (file.isFile() && file.getName().endsWith(".arff")) {
					System.out.println(file.getName().replace("arff", "csv"));
					// todo loop through each file
					arffFileName = file.getName();
					csvFileName = "mlp"+file.getName().replace("arff", "csv");
					trainPath = datasetPath + arffFolderName + "/"
							+ arffFileName;
					Instances data = null;
					try {
						data = new Instances(new BufferedReader(new FileReader(
								trainPath)));
					} catch (FileNotFoundException e1) {

						e1.printStackTrace();
					} catch (IOException e1) {

						e1.printStackTrace();
					}
					MultilayerPerceptron mlp = new MultilayerPerceptron();
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
					try {
						mlp.buildClassifier(data);
						mlp.setGUI(true);
						mlp.setLearningRate(0.5);
						mlp.setOptions(Utils.splitOptions("-L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H 4"));
						
					} catch (Exception e1) {
						// TODO Auto-generated catch block
						e1.printStackTrace();
					}

					if (data.classIndex() == -1) {
						data.setClassIndex(data.numAttributes() - 1);
					}
					int n = data.numInstances(), m = data.numAttributes();

					for (int attributeIndex = 1; attributeIndex < m; attributeIndex++) {
						if (data.attribute(attributeIndex).name() != null)
							currentAttributeName = data
									.attribute(attributeIndex).name()
									.replace("Hourly", "").replace("pro", "");
						else
							currentAttributeName = "";
						//TODO need to remove Month name from Current Attribute if it found...
						
						System.out.println(data.attribute(attributeIndex)
								.name());

						double rmsle = 0;
						double rmsleZero = 0;

						outputPath = datasetPath + csvFolderName + "/"
								+ currentAttributeName + csvFileName;

						try {
							// Create file
							FileWriter fstream = new FileWriter(outputPath);
							BufferedWriter out = new BufferedWriter(fstream);

							// int attributeindextoPredict = 1; // Last station
							// int attributeindextoPredict =2; //second station
							// int attributeindextoPredict =3; //first station
							double difference = 0;

							double rmslepred = 0;
							double rmsleact = 0;
							out.write("Instanceid,actual,predicted,difference");
							out.newLine();
							for (int curInstance = 0; curInstance < n; ++curInstance) {
								Instance t = data.instance(curInstance);
								double pred = mlp.classifyInstance(t), act = t
										.value(m - attributeIndex);
								
								BigDecimal pred1 = new BigDecimal(
										String.valueOf(pred)).setScale(2,
										BigDecimal.ROUND_HALF_UP);
								// difference = act - pred;
								difference = act - pred1.doubleValue();
								BigDecimal difference1 = new BigDecimal(
										String.valueOf(difference)).setScale(2,
										BigDecimal.ROUND_HALF_UP);

								rmslepred = (pred < 0 ? 0 : pred);
								rmsleact = (act < 0 ? 0 : act);
								// rmsle += (Math.log(pred + 1) - Math.log(act +
								// 1))
								// *
								// (Math.log(pred + 1) - Math.log(act + 1));
								// The above formula gives error if pred
								// variables
								// less
								// <0 so modify pred to be zero on this
								// instances
								rmsle += (Math.log(rmslepred + 1) - Math
										.log(rmsleact + 1))
										* (Math.log(rmslepred + 1) - Math
												.log(rmsleact + 1));
								rmsleZero += Math.log(rmsleact + 1)
										* Math.log(rmsleact + 1);
								// difference = act - pred;
								out.write((curInstance + 1) + "," + act + ","
										+ pred1.doubleValue() + ","
										+ difference1);
								out.newLine();

								// System.out.println(act+":"+pred+":"+rmsle);

							}
							out.close();
							rmsle = Math.sqrt(rmsle / n);
							rmsleZero = Math.sqrt(rmsleZero / n);
							System.out
									.println("Successfully written prediction to a file  "
											+ outputPath);
							System.out.println("# of training data: "
									+ data.numInstances());
							System.out.println("RMSLE on testing data: "
									+ rmsle);
							System.out.println("RMSLE on testing data Zero: "
									+ rmsleZero);
							
							WriteToOuputSummaryToLog(datasetPath + arffFolderName, String.format (
									arffFileName+",Multilayer Perceptron,"+currentAttributeName+","+data.numInstances()
									+","+outputPath+","
									+rmsle+","+rmsleZero));
							
						} catch (Exception e) {// Catch exception if any
							System.err.println("Error: " + e.getMessage());
						}

					}

				}

			}
		} catch (Exception e1) {

			e1.printStackTrace();
		}
	}

	public static void LinearRegressionPredictorClassifier(String datasetPath) {
		File folder = new File(datasetPath + arffFolderName);
		File[] listOfFiles = folder.listFiles();
		for (int i = 0; i < listOfFiles.length; i++) {
			currentAttributeName = null;
			outputPath = null;
			// if (i>=1) break;
			File file = listOfFiles[i];
			if (file.isFile() && file.getName().endsWith(".arff")) {
				System.out.println(file.getName().replace("arff", "csv"));
				// todo loop through each file
				arffFileName = file.getName();
				csvFileName = "lr"+file.getName().replace("arff", "csv");
				trainPath = datasetPath + arffFolderName + "/" + arffFileName;

				// read the data
				Instances data = null;
				try {
					data = new Instances(new BufferedReader(new FileReader(
							trainPath)));
				} catch (FileNotFoundException e1) {

					e1.printStackTrace();
				} catch (IOException e1) {

					e1.printStackTrace();
				}

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

				try {
					predictor.buildClassifier(data);
				} catch (Exception e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				}

				if (data.classIndex() == -1) {
					data.setClassIndex(data.numAttributes() - 1);
				}

				int n = data.numInstances(), m = data.numAttributes();

				for (int attributeIndex = 1; attributeIndex < m; attributeIndex++) {
					if (data.attribute(attributeIndex).name() != null)
						currentAttributeName = data.attribute(attributeIndex)
								.name().replace("Hourly", "")
								.replace("pro", "");
					else
						currentAttributeName = "";
					System.out.println(data.attribute(attributeIndex).name());

					double rmsle = 0;
					double rmsleZero = 0;

					outputPath = datasetPath + csvFolderName + "/"
							+ currentAttributeName + csvFileName;

					try {
						// Create file
						FileWriter fstream = new FileWriter(outputPath);
						BufferedWriter out = new BufferedWriter(fstream);

						// int attributeindextoPredict = 1; // Last station
						// int attributeindextoPredict =2; //second station
						// int attributeindextoPredict =3; //first station
						double difference = 0;

						double rmslepred = 0;
						double rmsleact = 0;
						out.write("Instanceid,actual,predicted,difference");
						out.newLine();
						for (int curInstance = 0; curInstance < n; ++curInstance) {
							Instance t = data.instance(curInstance);
							double pred = predictor.classifyInstance(t), act = t
									.value(m - attributeIndex);
							BigDecimal pred1 = new BigDecimal(
									String.valueOf(pred)).setScale(2,
									BigDecimal.ROUND_HALF_UP);
							// difference = act - pred;
							difference = act - pred1.doubleValue();
							BigDecimal difference1 = new BigDecimal(
									String.valueOf(difference)).setScale(2,
									BigDecimal.ROUND_HALF_UP);

							rmslepred = (pred < 0 ? 0 : pred);
							rmsleact = (act < 0 ? 0 : act);
							// rmsle += (Math.log(pred + 1) - Math.log(act + 1))
							// *
							// (Math.log(pred + 1) - Math.log(act + 1));
							// The above formula gives error if pred variables
							// less
							// <0 so modify pred to be zero on this instances
							rmsle += (Math.log(rmslepred + 1) - Math
									.log(rmsleact + 1))
									* (Math.log(rmslepred + 1) - Math
											.log(rmsleact + 1));
							rmsleZero += Math.log(rmsleact + 1)
									* Math.log(rmsleact + 1);
							// difference = act - pred;
							out.write((curInstance + 1) + "," + act + ","
									+ pred1.doubleValue() + "," + difference1);
							out.newLine();

							// System.out.println(act+":"+pred+":"+rmsle);

						}
						out.close();
						rmsle = Math.sqrt(rmsle / n);
						rmsleZero = Math.sqrt(rmsleZero / n);
						System.out
								.println("Successfully written prediction to a file  "
										+ outputPath);
						System.out.println("# of training data: "
								+ data.numInstances());
						System.out.println("RMSLE on testing data: " + rmsle);
						System.out.println("RMSLE on testing data Zero: "
								+ rmsleZero);
						WriteToOuputSummaryToLog(datasetPath + arffFolderName, String.format (
								arffFileName+",Linear regression,"+currentAttributeName+","+data.numInstances()
								+","+outputPath+","
								+rmsle+","+rmsleZero));
					} catch (Exception e) {// Catch exception if any
						System.err.println("Error: " + e.getMessage());
					}
				}
			}
		}

	}
	public static void WriteToOuputSummaryToLog(String filePath,String outputSummary)
	{
		try(PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(filePath+"/"+"output.csv", true)))) {
		    out.println(outputSummary);
		}catch (IOException e) {
		    //exception handling left as an exercise for the reader
		}
	}
	
	public static void main(String[] args) throws Exception {

		String datasetpath = "C:/CIT/Mini-Thesis/temperatureOnly/Processed/";
		System.out.println("------------------------------------------------Linear Regression-----------------------------------------------------------------");
		LinearRegressionPredictorClassifier(datasetpath);
		System.out.println("------------------------------------------------Multilayer Perceptron-----------------------------------------------------------------");
		MultiLayerPerceptroPredictorClassifier(datasetpath);
		System.out.println("------------------------------------------------Completed-----------------------------------------------------------------");
		// System.out.println("RMSLE on testing data: " + rmsle);
		// System.out.println("RMSLE on testing data Zero: " + rmsleZero);

	}
}