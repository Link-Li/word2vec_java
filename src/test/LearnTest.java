package test;


import vec.Learn;

import java.io.File;
import java.io.IOException;
import java.util.Date;

public class LearnTest {

    public static void main(String[] args) throws IOException {
        long timeStart = System.currentTimeMillis();
        Boolean isCbow = Boolean.TRUE;
        Learn learn = new Learn(isCbow, null, null, null, null);
        learn.learnFile(new File("corpus.txt"));
        learn.saveModel(new File("Cbow_model.bin"));
        long timeStop = System.currentTimeMillis();
        System.out.println("用时：" + (timeStop - timeStart)/1000 + "秒\n");


         timeStart = System.currentTimeMillis();
        isCbow = Boolean.FALSE;
        learn = new Learn(isCbow, null, null, null, null);
        learn.learnFile(new File("corpus.txt"));
        learn.saveModel(new File("SkipGram_model.bin"));
         timeStop = System.currentTimeMillis();
        System.out.println("用时：" + (timeStop - timeStart)/1000 + "秒\n");
    }
}
