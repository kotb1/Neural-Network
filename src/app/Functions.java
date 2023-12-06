package app;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class Functions 
{
	//public ArrayList <Integer> HiddenWeight = new ArrayList <Integer>();
	//public ArrayList <Integer> OutputWeight = new ArrayList <Integer>();
	//public  double [][] HiddenWeight={{0.3,-0.9,1},{-1.2,1,1}};
	//public  double [] OutputWeight= {1,0.8};
	public double [][] HiddenWeight;
	public double [] OutputWeight;
	public int Number_Of_Hidden_Neurons;
	public int Number_Of_Inputs;
	Functions(int Number_Of_Hidden_Neurons,int Number_Of_Inputs)
	{
		this.Number_Of_Hidden_Neurons=Number_Of_Hidden_Neurons;
		this.Number_Of_Inputs=Number_Of_Inputs;
		HiddenWeight = new double [Number_Of_Hidden_Neurons][Number_Of_Inputs];
		for(int i=0;i<Number_Of_Hidden_Neurons;i++) 
		{
			for(int j=0;j<Number_Of_Inputs;j++) 
			{
				Random rn = new Random();
				//int weight = rn.nextInt(10) + 1;
				double max=1.0f;
				double min=-1.0f;
				double weight = min+(max-min)*rn.nextDouble();
				HiddenWeight[i][j]=weight;
			}
		}
		OutputWeight= new double [Number_Of_Hidden_Neurons];
		for(int i=0;i<Number_Of_Hidden_Neurons;i++) 
		{
			Random rn = new Random();
			//int weight = rn.nextInt(10) + 1;
			double max=1.0f;
			double min=-1.0f;
			double weight = min+(max-min)*rn.nextDouble();
			OutputWeight[i]=weight;
		}
	}
	public void train(ArrayList <ImageData> m) 
	{
		for(int b =0;b<9000;b++) 
		{
		for(int i =0;i<m.size();i++) 
		{
			ArrayList <Double> sum = new ArrayList();
			for(int l =0;l<HiddenWeight.length;l++)
			{
				
				double sum1 =0;
				for(int j=0;j<m.get(i).pixels.length;j++) 
				{
					sum1 +=m.get(i).pixels[j]*HiddenWeight[l][j];
				}
				//System.out.println(sum);
				double weight = sigmoid(sum1);
				sum.add(weight);
				//System.out.println(weight);
				
				
				//System.out.println(output);
				
				//System.out.println(delta_output);
			}
			double output=0;
			for(int h=0;h<sum.size();h++) 
			{
				output+=sum.get(h)*OutputWeight[h];
			}
			output=sigmoid(output);
			double delta_output=output-m.get(i).label;
			delta_output*=output;
			delta_output*=1-output;
			//System.out.println(delta_output);
			ArrayList <Double>delta_outputs = new ArrayList();
			for(int q=0;q<OutputWeight.length;q++) 
			{
				double delta=delta_output*OutputWeight[q];
				delta*=sum.get(q);
				delta*=(1-sum.get(q));
				delta_outputs.add(delta);
			}
			double constant=0.3;
			for(int n=0;n<OutputWeight.length;n++) 
			{
				double result=delta_output*sum.get(n)*constant;
				OutputWeight[n]=OutputWeight[n]-result;
				//System.out.println(sum.get(n));
			}
			//System.out.println(OutputWeight[0]);
			//System.out.println(OutputWeight[1]);
			/*for(int n=0;n<HiddenWeight.length;n++) 
			{
				for(int q=0;q<HiddenWeight[n].length;q++) 
				{
					System.out.print(HiddenWeight[n][q]+",");
				}
				System.out.println();
			}*/
			for(int n=0;n<HiddenWeight.length;n++) 
			{
				for(int q=0;q<HiddenWeight[n].length;q++) 
				{
					double result=constant*delta_outputs.get(n)*m.get(i).pixels[q];
					HiddenWeight[n][q]=HiddenWeight[n][q]-result;
				}
			}
			/*for(int n=0;n<HiddenWeight.length;n++) 
			{
				for(int q=0;q<HiddenWeight[n].length;q++) 
				{
					System.out.print(HiddenWeight[n][q]+",");
				}
				System.out.println();
			}*/
		}
		}
	}
	public Double predict(ArrayList <ImageData> m) 
	{
		ArrayList <Double> output = new ArrayList();
		double result=-1;
		for(int i =0;i<m.size();i++) 
		{
			ArrayList <Double> sum = new ArrayList();
			for(int l =0;l<HiddenWeight.length;l++)
			{
				
				double sum1 =0;
				for(int j=0;j<m.get(i).pixels.length;j++) 
				{
					sum1 +=m.get(i).pixels[j]*HiddenWeight[l][j];
				}
				//System.out.println(sum);
				double weight = sigmoid(sum1);
				sum.add(weight);
			}
			double output1=0;
			for(int h=0;h<sum.size();h++) 
			{
				output1+=sum.get(h)*OutputWeight[h];
			}
			output1=sigmoid(output1);
			output.add(output1);
			//double delta_output=output1-m.get(i).label;
			//delta_output*=output1;
			//delta_output*=1-output1;
		}
		ArrayList <Integer> outputs = new ArrayList();
		for(int i=0;i<output.size();i++) 
		{
			outputs.add((int) Math.round(output.get(i)));
		}
		int acc = 0;
		for(int i =0;i<outputs.size();i++) 
		{
			if(m.get(i).label == outputs.get(i)) 
			{
				acc++;
			}
			
		}
		result=acc;
		result/=m.size();
		result*=100;
		return result;
	}
	public void predict(int[] sampleImgFeatures) 
	{
		ArrayList <Double> sum = new ArrayList();
		for(int l =0;l<HiddenWeight.length;l++)
		{
			
			double sum1 =0;
			for(int j=0;j<sampleImgFeatures.length;j++) 
			{
				sum1 +=sampleImgFeatures[j]*HiddenWeight[l][j];
			}
			double weight = sigmoid(sum1);
			sum.add(weight);
		}
		double output1=0;
		for(int h=0;h<sum.size();h++) 
		{
			output1+=sum.get(h)*OutputWeight[h];
		}
		output1=sigmoid(output1);
		System.out.println(output1);
		System.out.println(result(output1));
		
	}
	public String result(double o) 
	{
		String output=null;
		if(o>1) 
		{
			output="It is a dog";
		}
		else if(o<0) 
		{
			output="It is a cat";
		}
		else if(o<1 && o>0) 
		{
			double res1 =1-o;
			if(res1<o) 
			{
				output="It is a dog";
			}
			else
				output="It is a cat";
		}
		else if(o == 1) 
		{
			output="It is a dog";
		}
		else if( o == 0 ) 
		{
			output="It is a cat";
		}
		return output;
	}
	public void save() throws IOException 
	{
		String fileName="model.txt";
		FileOutputStream fos = new FileOutputStream(fileName, true);
	    fos.write("Hidden Layer Weights\r\n".getBytes());
	    for(int n=0;n<HiddenWeight.length;n++) 
		{
	    	fos.write((n+1+"\r\n").getBytes());
	    	String h = "";
	    	for(int q=0;q<HiddenWeight[n].length;q++) 
			{
				//fos.write((HiddenWeight[n][q]+",").getBytes());
	    		h+=HiddenWeight[n][q]+",";
			}
	    	//String content = new Scanner(new File("model.txt")).useDelimiter("\\Z").next();
			//String withoutLastCharacter = content.substring(0, content.length() - 1);
			//fos.write(withoutLastCharacter.getBytes());
	    	h=h.substring(0,h.length()-1);
	    	fos.write(h.getBytes());
			fos.write("\r\n".getBytes());
			fos.write("\r\n".getBytes());
		}
	    fos.write("Output Layer Weights\r\n".getBytes());
	    for(int i =0;i<OutputWeight.length;i++) 
	    {
	    	fos.write((i+1+"\r\n").getBytes());
	    	fos.write((OutputWeight[i]+"").getBytes());
	    	fos.write("\r\n".getBytes());
			fos.write("\r\n".getBytes());
	    }
	    fos.close();
	}
	public double sigmoid(double x) 
	{
		double result=1+Math.exp(-x);
		result =1/result;
		return result;
	}
	Functions(int Number_Of_Hidden_Neurons, int Number_Of_Inputs,String Filename) throws IOException
	{
		BufferedReader objReader = new BufferedReader(new FileReader(Filename));
		ArrayList <ArrayList<Double>> height_weights= new ArrayList <ArrayList<Double>>();
		ArrayList <Double> output_weights= new ArrayList <Double>();
		String strCurrentLine="";
		/*strCurrentLine = objReader.readLine();
		strCurrentLine = objReader.readLine();
		strCurrentLine = objReader.readLine();
		System.out.println(strCurrentLine);*/
		//System.out.println(objReader.readLine());
		boolean f = true;
		while ((strCurrentLine = objReader.readLine()) != null) {
		    //System.out.println(strCurrentLine);
			if((strCurrentLine = objReader.readLine())!=null) 
			{
				//System.out.println(strCurrentLine);
				//System.out.println(strCurrentLine);
				if(strCurrentLine.equals("Output Layer Weights")) 
				{
					strCurrentLine = objReader.readLine();
					strCurrentLine = objReader.readLine();
					output_weights.add(Double.valueOf(strCurrentLine));
					f=false;
					//System.out.println(strCurrentLine);
					//break;
				}
				else if (f)
				{
					//System.out.println(strCurrentLine);
					strCurrentLine = objReader.readLine();
					height_weights.add(split(strCurrentLine));
				}
				else if(!f) 
				{
					strCurrentLine = objReader.readLine();
					output_weights.add(Double.valueOf(strCurrentLine));
				}
			}
			else 
				break;
			
		}
		//System.out.println(height_weights.size());
		//System.out.println(output_weights.size());
		HiddenWeight = new double [Number_Of_Hidden_Neurons][Number_Of_Inputs];
		OutputWeight= new double [Number_Of_Hidden_Neurons];
		for(int i=0;i<height_weights.size();i++) 
		{
			for(int j=0;j<height_weights.get(i).size();j++) 
			{
				HiddenWeight[i][j] = height_weights.get(i).get(j);
			} 
		}
		for(int j=0;j<output_weights.size();j++) 
		{
			OutputWeight[j] = output_weights.get(j);
		} 
	}
	public static ArrayList <Double> split (String x)
	{
		ArrayList <Double> Strings = new ArrayList();
		int comma=x.indexOf(',');
		String f = x.substring(0,comma);
		double y = Double.valueOf(f);
		Strings.add(y);
		x=x.substring(comma);
		Strings=ff(x,Strings);
		return Strings;
	}
	public static ArrayList <Double> ff (String x,ArrayList <Double> Strings)
	{
		if(x.charAt(0) == ',') 
		{
			x=x.substring(1);
			int comma=x.indexOf(',');
			if(comma!=-1) 
			{
				String f = x.substring(0,comma);
				double y = Double.valueOf(f);
				Strings.add(y);
				x=x.substring(comma);
				ff(x,Strings);
			}
			else
			{
				double y = Double.valueOf(x);
				Strings.add(y);
			}
		}
		return Strings;
	}
}
