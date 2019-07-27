package vec;


import vec.domain.HiddenNeuron;
import vec.domain.Neuron;
import vec.domain.WordNeuron;
import vec.util.Haffman;
import vec.util.MapCount;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class Learn {

    /**
     * 用来存储词和词对应的一些信息，这些词是不重复的
     */
    private Map<String, Neuron> wordMap = new HashMap<>();
    /**
     * 训练多少个特征
     */
    private int layerSize = 200;

    /**
     * 上下文窗口大小
     */
    private int window = 5;

//    表示下采样阀值。
    private double sample = 1e-3;
//    表示学习率。skip模式下默认为0.025， cbow模式下默认是0.05。
    private double alpha = 0.025;
    private double startingAlpha = alpha;

    public int EXP_TABLE_SIZE = 1000;

    private Boolean isCbow = false;

    private double[] expTable = new double[EXP_TABLE_SIZE];

    private int trainWordsCount = 0;

//    这个应该是使用sigmoid函数的范围，在x取【-6,6】的范围内，sigmoid的反向传播才会有点意义，不然反向传播都是0了
    private int MAX_EXP = 6;

    public Learn(Boolean isCbow, Integer layerSize, Integer window, Double alpha, Double sample) {
        createExpTable();
        if (isCbow != null) {
            this.isCbow = isCbow;
        }
        if (layerSize != null)
            this.layerSize = layerSize;
        if (window != null)
            this.window = window;
        if (alpha != null)
            this.alpha = alpha;
        if (sample != null)
            this.sample = sample;
    }

    public Learn() {
        createExpTable();
    }

    /**
     * trainModel
     * 
     * @throws IOException
     */
    private void trainModel(File file) throws IOException {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {
            String temp = null;
            long nextRandom = 5;
//            用来计数目前分析了多少单词了
            int wordCount = 0;
//            用来标记上次分析的单词的数目
            int lastWordCount = 0;
            int wordCountActual = 0;
            while ((temp = br.readLine()) != null) {
                if (wordCount - lastWordCount > 10000) {
                    System.out.println("alpha:" + alpha + "\tProgress: "
                            + (int) (wordCountActual / (double) (trainWordsCount + 1) * 100) + "%");
                    wordCountActual += wordCount - lastWordCount;
                    lastWordCount = wordCount;
                    alpha = startingAlpha * (1 - wordCountActual / (double) (trainWordsCount + 1));
                    if (alpha < startingAlpha * 0.0001) {
                        alpha = startingAlpha * 0.0001;
                    }
                }
//                String[] strs = temp.split(" ");
                String[] strs = temp.split("[\\s　]+");// 修改，支持出现多个半角全角空格，制表符分隔。
                wordCount += strs.length;
                List<WordNeuron> sentence = new ArrayList<WordNeuron>();
                for (int i = 0; i < strs.length; i++) {
//                    获取上次读文件读出来的词对应的neuron信息(Z)
                    Neuron entry = wordMap.get(strs[i]);
                    if (entry == null) {
                        continue;
                    }
                    // The subsampling randomly discards frequent words while
                    // keeping the
                    // ranking same
//                    下采样随机丢弃频繁的单词，同时保持排名相同，随机跳过一些词的训练
                    if (sample > 0) {
                        double ran = (Math.sqrt(entry.freq / (sample * trainWordsCount)) + 1)
                                * (sample * trainWordsCount) / entry.freq;
                        nextRandom = nextRandom * 25214903917L + 11;
                        double a = (nextRandom & 0xFFFF) / (double) 65536;
//                        //频率越大的词，对应的ran就越小，越容易被抛弃，被跳过
                        if (ran < (nextRandom & 0xFFFF) / (double) 65536) {
                            continue;
                        }
                    }
                    sentence.add((WordNeuron) entry);
                }

                for (int index = 0; index < sentence.size(); index++) {
                    nextRandom = nextRandom * 25214903917L + 11;
                    if (isCbow) {
                        cbowGram(index, sentence, (int) nextRandom % window);
                    } else {
                        skipGram(index, sentence, (int) nextRandom % window);
                    }
                }

            }
            System.out.println("Vocab size: " + wordMap.size());
            System.out.println("Words in train file: " + trainWordsCount);
            System.out.println("sucess train over!");
        }
    }

    /**
     * skip gram 模型训练
     * 
     * @param sentence
     * @param b
     */
    private void skipGram(int index, List<WordNeuron> sentence, int b) {
        // TODO Auto-generated method stub
        WordNeuron word = sentence.get(index);
        int a, c = 0;
        for (a = b; a < window * 2 + 1 - b; a++) {
            if (a == window) {
                continue;
            }
            c = index - window + a;
            if (c < 0 || c >= sentence.size()) {
                continue;
            }

            double[] neu1e = new double[layerSize];// 误差项
            // HIERARCHICAL SOFTMAX
            List<Neuron> neurons = word.neurons;
            WordNeuron we = sentence.get(c);
            for (int i = 0; i < neurons.size(); i++) {
                HiddenNeuron out = (HiddenNeuron) neurons.get(i);
                double f = 0;
                // Propagate hidden -> output
                for (int j = 0; j < layerSize; j++) {
                    f += we.syn0[j] * out.syn1[j];
                }
                if (f <= -MAX_EXP || f >= MAX_EXP) {
                    continue;
                } else {
                    f = (f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2);
                    f = expTable[(int) f];
                }
                // 'g' is the gradient multiplied by the learning rate
                double g = (1 - word.codeArr[i] - f) * alpha;
                // Propagate errors output -> hidden
                for (c = 0; c < layerSize; c++) {
                    neu1e[c] += g * out.syn1[c];
                }
                // Learn weights hidden -> output
                for (c = 0; c < layerSize; c++) {
                    out.syn1[c] += g * we.syn0[c];
                }
            }

            // Learn weights input -> hidden
            for (int j = 0; j < layerSize; j++) {
                we.syn0[j] += neu1e[j];
            }
        }

    }

    /**
     * 词袋模型
     * 
     * @param index
     * @param sentence
     * @param b
     */
    private void cbowGram(int index, List<WordNeuron> sentence, int b) {
        WordNeuron word = sentence.get(index);
        int a, c = 0;

        List<Neuron> neurons = word.neurons;
        double[] neu1e = new double[layerSize];// 误差项
        double[] neu1 = new double[layerSize];// 误差项
        WordNeuron last_word;

        for (a = b; a < window * 2 + 1 - b; a++)
            if (a != window) {
                c = index - window + a;
                if (c < 0)
                    continue;
                if (c >= sentence.size())
                    continue;
                last_word = sentence.get(c);
                if (last_word == null)
                    continue;
                for (c = 0; c < layerSize; c++)
                    neu1[c] += last_word.syn0[c];
            }

        // HIERARCHICAL SOFTMAX
        for (int d = 0; d < neurons.size(); d++) {
            HiddenNeuron out = (HiddenNeuron) neurons.get(d);
            double f = 0;
            // Propagate hidden -> output
            for (c = 0; c < layerSize; c++)
                f += neu1[c] * out.syn1[c];
//            超出设置的sigmoid范围的数据都舍弃掉
            if (f <= -MAX_EXP)
                continue;
            else if (f >= MAX_EXP)
                continue;
            else{
//                double ppppp = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2));
                f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            }

            // 'g' is the gradient multiplied by the learning rate
             double g = (1 - word.codeArr[d] - f) * alpha;
            // double g = f*(1-f)*( word.codeArr[i] - f) * alpha;
//            double g = f * (1 - f) * (word.codeArr[d] - f) * alpha;
            //
            for (c = 0; c < layerSize; c++) {
                neu1e[c] += g * out.syn1[c];
            }
            // Learn weights hidden -> output
            for (c = 0; c < layerSize; c++) {
                out.syn1[c] += g * neu1[c];
            }
        }
        for (a = b; a < window * 2 + 1 - b; a++) {
            if (a != window) {
                c = index - window + a;
                if (c < 0)
                    continue;
                if (c >= sentence.size())
                    continue;
                last_word = sentence.get(c);
                if (last_word == null)
                    continue;
                for (c = 0; c < layerSize; c++)
                    last_word.syn0[c] += neu1e[c];
            }

        }
    }

    /**
     * 统计词频
     * 
     * @param file
     * @throws IOException
     */
    private void readVocab(File file) throws IOException {
        MapCount<String> mc = new MapCount<>();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {
            String temp = null;
            while ((temp = br.readLine()) != null) {
                String[] split = temp.split("[\\s　]+");// 修改，支持出现多个半角全角空格，制表符分隔。
                trainWordsCount += split.length;
                for (String string : split) {
                    mc.add(string);
                }
            }
        }
        for (Entry<String, Integer> element : mc.get().entrySet()) {
            wordMap.put(element.getKey(),
                    new WordNeuron(element.getKey(), (double) element.getValue() / mc.size(), layerSize));
        }
    }

    /**
     * 对文本进行预分类
     * 
     * @param files
     * @throws IOException
     * @throws FileNotFoundException
     */
    private void readVocabWithSupervised(File[] files) throws IOException {
        for (int category = 0; category < files.length; category++) {
            // 对多个文件学习
            MapCount<String> mc = new MapCount<>();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(files[category])))) {
                String temp = null;
                while ((temp = br.readLine()) != null) {
                    String[] split = temp.split("[\\s　]+");
                    trainWordsCount += split.length;
                    for (String string : split) {
                        mc.add(string);
                    }
                }
            }
            for (Entry<String, Integer> element : mc.get().entrySet()) {
                double tarFreq = (double) element.getValue() / mc.size();
                if (wordMap.get(element.getKey()) != null) {
                    double srcFreq = wordMap.get(element.getKey()).freq;
                    if (srcFreq >= tarFreq) {
                        continue;
                    } else {
                        Neuron wordNeuron = wordMap.get(element.getKey());
                        wordNeuron.category = category;
                        wordNeuron.freq = tarFreq;
                    }
                } else {
                    wordMap.put(element.getKey(), new WordNeuron(element.getKey(), tarFreq, category, layerSize));
                }
            }
        }
    }


    /**
     * 在训练过程中需要用到大量的sigmoid值计算，如果每次都临时去算exp(x)的值，将会影响性能；
     * 当对精度的要求不是很严格的时候，我们可以采用近似的运算。
     * 在word2vec中，将区间[-MAX_EXP, MAX_EXP](代码中MAX_EXP默认值为6)等距划分为EXP_TABLE_SIZE等份，
     * 并将每个区间的sigmoid值计算好存入到expTable中。在需要使用时，只需要确定所属的区间，属于哪一份，然后直接去数组中查找。
     * expTable初始化代码如下:
     */
    private void createExpTable() {
        for (int i = 0; i < EXP_TABLE_SIZE; i++) {
            expTable[i] = Math.exp((((i / (double) EXP_TABLE_SIZE) * 2 - 1) * MAX_EXP));
            expTable[i] = expTable[i] / (expTable[i] + 1); // 把原来的sigmoid的函数进行了一下变化，把符号提取出来之后的形式
        }
    }

    /**
     * 根据文件学习
     * 
     * @param file
     * @throws IOException
     */
    public void learnFile(File file) throws IOException {
//      读取文件中的信息，并将词不重复的放到wordMap中
        readVocab(file);
//        对wordMap进行处理，构建一个哈弗曼树，
        new Haffman(layerSize).make(wordMap.values());

        // 查找每个神经元
//        确定每个neuron的位置，使用makeNeurons方法
        for (Neuron neuron : wordMap.values()) {
            ((WordNeuron) neuron).makeNeurons();
        }

        trainModel(file);
    }

    /**
     * 根据预分类的文件学习
     * 
     * @param summaryFile
     *            合并文件
     * @param classifiedFiles
     *            分类文件
     * @throws IOException
     */
    public void learnFile(File summaryFile, File[] classifiedFiles) throws IOException {
        readVocabWithSupervised(classifiedFiles);
        new Haffman(layerSize).make(wordMap.values());
        // 查找每个神经元
        for (Neuron neuron : wordMap.values()) {
            ((WordNeuron) neuron).makeNeurons();
        }
        trainModel(summaryFile);
    }

    /**
     * 保存模型
     */
    public void saveModel(File file) {
        try (DataOutputStream dataOutputStream = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(file)))) {
            dataOutputStream.writeInt(wordMap.size());
            dataOutputStream.writeInt(layerSize);
            double[] syn0 = null;
            for (Entry<String, Neuron> element : wordMap.entrySet()) {
                dataOutputStream.writeUTF(element.getKey());
                syn0 = ((WordNeuron) element.getValue()).syn0;
                for (double d : syn0) {
                    dataOutputStream.writeFloat(((Double) d).floatValue());
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public int getLayerSize() {
        return layerSize;
    }

    public void setLayerSize(int layerSize) {
        this.layerSize = layerSize;
    }

    public int getWindow() {
        return window;
    }

    public void setWindow(int window) {
        this.window = window;
    }

    public double getSample() {
        return sample;
    }

    public void setSample(double sample) {
        this.sample = sample;
    }

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
        this.startingAlpha = alpha;
    }

    public Boolean getIsCbow() {
        return isCbow;
    }

    public void setIsCbow(Boolean isCbow) {
        this.isCbow = isCbow;
    }

}
