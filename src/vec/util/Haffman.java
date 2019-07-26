package vec.util;

import vec.domain.HiddenNeuron;
import vec.domain.Neuron;

import java.util.Collection;
import java.util.TreeSet;

/**
 * 构建Haffman编码树
 * 
 * @author ansj
 *
 */
public class Haffman {
  private int layerSize;

  public Haffman(int layerSize) {
    this.layerSize = layerSize;
  }

//  就是一个排序的二叉树，这里需要Neuron类中的compareTo方法来确保排序的方式
  private TreeSet<Neuron> set = new TreeSet<>();

  public void make(Collection<Neuron> neurons) {
    set.addAll(neurons);
    while (set.size() > 1) {
      merger();
    }
  }

    /**
     * 根据传入的neurons，对这些词进行构建哈弗曼树，这里是提取两个节点，然后融合成一个树，
     * 根据上面的make，一个循环下来，就将所有的节点都融合到了树里面
     */
  private void merger() {
    HiddenNeuron hn = new HiddenNeuron(layerSize);
    Neuron min1 = set.pollFirst();
    Neuron min2 = set.pollFirst();
    hn.category = min2.category;
    hn.freq = min1.freq + min2.freq;
    min1.parent = hn;
    min2.parent = hn;
    min1.code = 0;
    min2.code = 1;
//    将新创建的节点加入到树中，然后可以根据这个进行下一个树的构建
//      这里说的树是指两个节点合成一个树
    set.add(hn);
  }

}
