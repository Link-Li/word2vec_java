package vec.domain;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class WordNeuron extends Neuron {
  public String name;
  public double[] syn0 = null; // input->hidden
//  这里可能指的是根据父节点来找所谓的这个叶子节点，这个里面存的都是父节点，
//  最开始的是按照从小到大的方式存的父节点，然后之后进行了一下反转，第一个存的是最终的顶级父节点
  public List<Neuron> neurons = null;// 路径神经元
//  这里记录的是neurons中的父节点的位置，是左子树，还是右子树，但是这个没有记录最开始的那个顶级父节点，而且最后记录了一下这个节点的是左子树还是右子树
  public int[] codeArr = null;

  /**
   * 感觉这里好像是根据传入的neuron，然后查找这个neuron在整个哈弗曼树中的位置，
   * 记录这个neuron的所有的父节点，以及父节点在每个子树中的位置（左右子树）。最后再记录一下这个neuron是左还是右子树
   * @return
   */
  public List<Neuron> makeNeurons() {
    if (neurons != null) {
      return neurons;
    }
    Neuron neuron = this;
    neurons = new LinkedList<>();
//    将一个叶子节点的所有的父节点找出来，最后一个等于null的就是最大的父节点
    while ((neuron = neuron.parent) != null) {
      neurons.add(neuron);
    }
    Collections.reverse(neurons);
    codeArr = new int[neurons.size()];

    for (int i = 1; i < neurons.size(); i++) {
      codeArr[i - 1] = neurons.get(i).code;
    }
    codeArr[codeArr.length - 1] = this.code;

    return neurons;
  }

  public WordNeuron(String name, double freq, int layerSize) {
    this.name = name;
    this.freq = freq;
    this.syn0 = new double[layerSize];
    Random random = new Random();
    for (int i = 0; i < syn0.length; i++) {
      syn0[i] = (random.nextDouble() - 0.5) / layerSize;
    }
  }

  /**
   * 用于有监督的创造hoffman tree
   * 
   * @param name
   * @param freq
   * @param layerSize
   */
  public WordNeuron(String name, double freq, int category, int layerSize) {
    this.name = name;
    this.freq = freq;
    this.syn0 = new double[layerSize];
    this.category = category;
    Random random = new Random();
    for (int i = 0; i < syn0.length; i++) {
      syn0[i] = (random.nextDouble() - 0.5) / layerSize;
    }
  }

}