package test;


import vec.Learn;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.logging.SimpleFormatter;

public class LearnTest {

    public static void main(String[] args) throws IOException {
//
//        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss");
//        long timeStart = System.currentTimeMillis();
//        Boolean isCbow = Boolean.TRUE;
//        Learn learn = new Learn(isCbow, null, null, null, null);
//        learn.learnFile(new File("corpus.txt"));
//        learn.saveModel(new File(simpleDateFormat.format(new Date()) + "Cbow_model.bin"));
//        long timeStop = System.currentTimeMillis();
//        System.out.println("用时：" + (timeStop - timeStart)/1000 + "秒\n");
//
//
//         timeStart = System.currentTimeMillis();
//        isCbow = Boolean.FALSE;
//        learn = new Learn(isCbow, null, null, null, null);
//        learn.learnFile(new File("corpus.txt"));
//        learn.saveModel(new File(simpleDateFormat.format(new Date()) + "SkipGram_model.bin"));
//         timeStop = System.currentTimeMillis();
//        System.out.println("用时：" + (timeStop - timeStart)/1000 + "秒\n");


        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss");

        long timeStart = System.currentTimeMillis();
        Boolean isCbow = Boolean.FALSE;
        Learn learn = new Learn(isCbow, null, null, null, null);
        learn.learnFile(new File("corpus.txt"));
        learn.saveModel(new File(simpleDateFormat.format(new Date()) + "Cbow_model.bin"));
        long timeStop = System.currentTimeMillis();
        System.out.println("用时：" + (timeStop - timeStart)/1000 + "秒\n");
    }
}
