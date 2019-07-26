package vec.domain;

public abstract class Neuron implements Comparable<Neuron> {
//  目前来看，这个字段存的是这个单词的频率
  public double freq;
//  好像是确定哈弗曼树中的父节点的
  public Neuron parent;
//  好像是用来分左右子树的，就是构建哈夫曼树的时候，用0和1来进行区分
  public int code;
  // 语料预分类
  public int category = -1;

  @Override
  public int compareTo(Neuron neuron) {
    if (this.category == neuron.category) {
      if (this.freq > neuron.freq) {
        return 1;
      } else {
        return -1;
      }
    } else if (this.category > neuron.category) {
      return 1;
    } else {
      return -1;
    }
  }
}
