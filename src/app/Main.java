package app;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class Main {
    public static void main(String[] args) throws IOException, Exception {
        //Load Data
        File[] images= new File("E:\\Cs7\\Soft Computing\\assignment4\\Cats&DogsSampleDataset").listFiles();
        //System.out.println(images[1]);
        ImageData[] data = new ImageData[images.length];
        for (int i = 0; i < images.length; i++) {
            data[i]=new ImageData();
        	data[i].setPixels(ImageHandler.ImageToIntArray(images[i]));
            data[i].setLabel(images[i].getName().contains("cat")? 0 : 1);
        }
        PrintWriter writer = new PrintWriter("model.txt");
        writer.print("");
        // other operations
        writer.close();
        /*for(int i =0;i<data.length;i++) 
        {
        	for(int j=0;j<data[i].pixels.length;j++) 
        	{
        		System.out.println(data[i].pixels[j]);
        	}
        	System.out.println(data[i].label);
        }*/
        //Shuffle
        List<ImageData> tempData = Arrays.asList(data);
        Collections.shuffle(tempData);
        tempData.toArray(data);
        //tempData.clear();

        //Split the data into training (75%) and testing (25%) sets
        /*int[][] trainingSetFeatures, testingSetFeatures;
        int[] trainingSetLabels, testingSetLabels;
        trainingSetFeatures = new int [30][];
        testingSetFeatures = new int [10][];
        trainingSetLabels=new int[30];*/
        int percentage = (int) (0.75 * data.length);
        //System.out.println(percentage);
        ArrayList <ImageData>training_Set = new ArrayList<ImageData>();
        ArrayList <ImageData>testing_Set = new ArrayList<ImageData>();
        for(int i =0;i<images.length;i++) 
        {
        	if(i<percentage) 
        	{
        		training_Set.add(data[i]);
        	}
        	else
        		testing_Set.add(data[i]);
        }
        
        /*for(int i=0;i<data.length;i++) 
        {
        	System.out.println(data[i].label);
        }
        System.out.println("train:");
        for(int i=0;i<training_Set.size();i++) 
        {
        	System.out.println(training_Set.get(i).label);
        }
        System.out.println("test:");
        for(int i=0;i<testing_Set.size();i++) 
        {
        	System.out.println(testing_Set.get(i).label);
        }*/
        /*
            ...
         */

        //Create the NN
       // NeuralNetwork nn = new NeuralNetwork();
        Functions nn = new Functions(30,training_Set.get(0).pixels.length);
        //Set the NN architecture
        /*
            ...
         */

        //Train the NN
       // nn.train(trainingSetFeatures, trainingSetLabels);
        nn.train(training_Set);
        
        //Test the model
       // int[] predictedLabels = nn.predict(testingSetFeatures);
        //double accuracy = nn.calculateAccuracy(predictedLabels, testingSetLabels);
        double accuracy=nn.predict(testing_Set);
        System.out.println(accuracy);
        
        //Save the model (final weights)
       // nn.save("model.txt");
        nn.save();
        
        //Load the model and use it on an image
        //NeuralNetwork nn2 = NeuralNetwork.load("model.txt");
        Functions nn2 =  new Functions(nn.Number_Of_Hidden_Neurons,nn.Number_Of_Inputs,"model.txt");
        int[] sampleImgFeatures = ImageHandler.ImageToIntArray(new File("E:\\Cs7\\Soft Computing\\assignment4\\Cats&DogsSampleDataset\\dogs_00001.jpg"));
       // int samplePrediction = nn2.predict(sampleImgFeatures);
        nn2.predict(sampleImgFeatures);
        ImageHandler.showImage("E:\\Cs7\\Soft Computing\\assignment4\\Cats&DogsSampleDataset\\dogs_00001.jpg");
        //Print "Cat" or "Dog"
        /*
            ...
         */
    }
}