# KNN优化
- 简介
	- 最近一个CV的论文看到作者使用了Ball tree结构的近邻算法，加上很久没有写关于传统机器学习算法的内容了，这里稍微介绍一下近邻算法的优化方法。
	- 一般而言，除了Brute Force这种高复杂度方法，目前的近邻算法优化方式主要两种，即K-D tree、Ball tree，这两种方法都是基于查询数据结构的优化（**也就是邻居搜索方式的优化**）。
	- 本案例使用鸢尾花数据集，且本案例只重点关注搜索部分，KNN原理简单，可以查看[我之前的博客](https://blog.csdn.net/zhouchen1998/article/details/84651435)。
- 算法缺陷
	- KNN是基于实例的学习，其核心思想是当前样本的标签为训练样本中最接近它的样本的标签，也就是说模型需要存储所有训练集，这**非常耗费存储空间**。
	- 由于需要判断类别的M个样本，每个都要与N个训练样本计算距离（若特征D维），这会使得计算量大。$O(N*M*D)$
	- 由于基于特征距离计算类别，对特征的数据尺度非常敏感。
- 代码说明
	- 本篇文章重点在于理论叙述而不是编码，所以均使用sklearn封装模块完成编码，具体参考[官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html)。
- Brute Force
	- 计算复杂度
		- 对于N个样本在D维特征空间，可以认为计算复杂度为$O(D*N)$。
	- 代码
		- ```python
			def test_accuracy():
			    """
			    测试邻居寻找准确率
			    :return:
			    """
			    x_train, y_train, x_test, y_test = get_dataset()
			    nnb = NearestNeighbors(n_neighbors=3, algorithm='brute').fit(x_train)
			    distance, index = nnb.kneighbors([x_test[0]])
			    print("nearest distance", distance)
			    print("nearest index", index)
			    print("nearest label", y_train[index])
			
			
			def test_time():
			    """
			    测试邻居查找时间
			    :return:
			    """
			    x_train, y_train, x_test, y_test = get_dataset()
			    nnb = NearestNeighbors(n_neighbors=3, algorithm='brute').fit(x_train)
			    start_time = time.time()
			    for sample in x_test:
			        distance, index = nnb.kneighbors([sample])
			    end_time = time.time()
			    print("Spend:{} s".format(end_time-start_time))
			```
	- 运行效果
		- 可以看到，准确率还不错，运行全部测试样本的计算，花费了0.0139秒。
		- ![](https://img-blog.csdnimg.cn/20190522202004978.png)
- K-D tree
	- 概念
		- K指邻居数目
		- D指特征维度
		- tree指二叉树
	- 原理
		- 如果我们知道A与B很接近、B与C很接近，那么可以假定A与C很接近，不用具体计算其距离。
		- 基于这种思想，利用二叉树结构进行邻居查询将是一种降低运算复杂度的不错选择。
	- 算法描述
		- 定义树节点的信息
			- 分裂方式(split_method)
			- 分裂点(split_point)
			- 左子树(left_tree)
			- 右子树(right_tree)
		- 定义树的生成方式（**树的生成过程就是数据空间的分割过程**）
			- 根据当前下标区间[L,R]（如5个样本则为[0-4]），计算样本集在每个特征维度上的方差，取方差最大的那一维d作为分裂方式，并且将所有样本按照d值进行排序，取中间的样本点mid为当前分裂节点的分裂点，然后，以生成的当前节点为父节点，[L,mid-1]和[mid+1,R]为左右子树建树。如此递归，直到区间只有一个样本点。
			- 经过这个过程，它将数据空间分割成很多小的空间，且每个空间只有一个样本点。
			- 举例
				- 举个例子，当特征只有两维，那么可视化啊就是一个平面被分割为了很多小的平面矩形，且每个矩形只包含一个样本点。
		- 定义树的查询方式
			- 查询，也就是要找这个样本点应该在的子空间而已。
			- 每次在二叉树上查找时，先看节点的分裂方式是什么，也就是分裂特征是多个特征中的哪一维的特征。如果测试样本在那个特征上的值小于分裂点的值，就在左子树上进行查询；如果大于分裂点的值，就在右子树上进行查询。
			- 子树查询结束，回溯到根节点，判断，以该点为圆心，目前得到的最小距离为半径，产生的圆区域是否相交于区间分裂那一维的平面，若相交，则需要查询分裂平面那一侧的子树，同时，判断能否用根节点到该点的距离更新最近距离。
			- 为什么呢
			- 所以，由于每个分裂节点都可能查询左右子树，查询的复杂度可能从$O(log(n))$变为$O( \sqrt {(n)})$。
	- 特点
		- 对参数空间沿着数据特征轴N进行划分，很高效，因为划分过程是在N上进行，而不用管样本维数D。
		- 只有当D很小(D<20)的时候，运算块，当D大即特征高维时，计算会慢不少。
	- 代码实现
		- ```python
			# -*-coding:utf-8-*-
			from sklearn.datasets import load_iris
			from sklearn.model_selection import train_test_split
			from sklearn.neighbors import NearestNeighbors
			import time
			
			
			def get_dataset():
			    """
			    获取数据集
			    :return:
			    """
			    data = load_iris()
			    x, y = data['data'], data['target']
			    x_train_, x_test_, y_train_, y_test_ = train_test_split(x, y, test_size=0.2, random_state=2019)  # 保证运行不变性
			    return x_train_, y_train_, x_test_, y_test_
			
			
			def test_accuracy():
			    """
			    测试准确率
			    :return:
			    """
			    x_train, y_train, x_test, y_test = get_dataset()
			    nnb = NearestNeighbors(n_neighbors=3, algorithm='kd_tree').fit(x_train)
			    distance, index = nnb.kneighbors([x_test[0]])
			    print("nearest distance", distance)
			    print("nearest index", index)
			    print("true label", y_test[0])
			    print("nearest label", y_train[index])
			
			
			def test_time():
			    """
			    测试邻居查找时间
			    :return:
			    """
			    x_train, y_train, x_test, y_test = get_dataset()
			    nnb = NearestNeighbors(n_neighbors=3, algorithm='kd_tree').fit(x_train)
			    start_time = time.time()
			    for sample in x_test:
			        distance, index = nnb.kneighbors([sample])
			    end_time = time.time()
			    print("Spend:{} s".format(end_time-start_time))
			
			
			if __name__ == '__main__':
			    test_accuracy()
			    test_time()
			```
	- 运行结果
		- ![](https://img-blog.csdnimg.cn/2019062020133577.png)
		- 可以看到，比起Brute，KDTree查找速度快了很多。
	- 计算复杂度
		- 最好$O(D*log(N))$，当D很大时效率不如Brute。
- Ball tree
	- 概述
		- 为了改进KDtree的二叉树树形结构，并且沿着笛卡尔坐标进行划分的低效率，Ball tree将在一系列相嵌套的超球体上分割数据。
		- 不同于KD Tree使用超矩形划分区域而是使用超球面划分。这导致在构建数据结构的花费上大于KDtree，但是在高维甚至很高维特征的数据上都表现的很高效。
	- 原理
		- Ball tree递归地将数据划分为由质心C和半径r定义的节点，使得节点中的每个点都位于由r和C定义的超球内。通过使用三角不等式来减少邻居搜索的候选点数量的$|x+y| \leq |x| + |y|$。
		- ![图片源于论文](https://img-blog.csdnimg.cn/20190620202131986.png)
	- 算法描述
		- 划分方式
			- 选择一个距离当前圆心最远的训练样本点$i_1$和距离$i_1$最远的训练样本点$i_2$，将圆中所有离这两个点最近的训练样本点都赋给这两个簇的中心，然后计算每一个簇的中心点和包含所有其所属训练样本点的最小半径。
			- 这各划分方式是线性的。
		- 查询方式
			- 使用Ball tree时，先自上而下找到包含目标的叶子结点$(c, r)$，从此结点中找到离它最近的训练样本点，这个距离就是最近邻的距离的上界。
			- 检查它的兄弟结点中是否包含比这个上界更小的训练样本点。检查方式为：如果目标点距离兄弟结点的圆心的距离 > 兄弟节点所在的圆半径 + 之前得到的上界的值，则这个兄弟结点不可能包含所要的训练样本点（可以理解为构成一个三角形，两圆必定相交）。否则，检查这个兄弟结点是否包含符合条件的训练样本点。
	- 计算复杂度
		- $O(D*log(N))$
	- 代码实现
		- ```python
			# -*-coding:utf-8-*-
			"""author: Zhou Chen
			   datetime: 2019/6/20 20:29
			   desc: the project
			"""
			# -*-coding:utf-8-*-
			from sklearn.datasets import load_iris
			from sklearn.model_selection import train_test_split
			from sklearn.neighbors import NearestNeighbors
			import time
			
			
			def get_dataset():
			    """
			    获取数据集
			    :return:
			    """
			    data = load_iris()
			    x, y = data['data'], data['target']
			    x_train_, x_test_, y_train_, y_test_ = train_test_split(x, y, test_size=0.2, random_state=2019)  # 保证运行不变性
			    return x_train_, y_train_, x_test_, y_test_
			
			
			def test_accuracy():
			    """
			    测试准确率
			    :return:
			    """
			    x_train, y_train, x_test, y_test = get_dataset()
			    nnb = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(x_train)
			    distance, index = nnb.kneighbors([x_test[0]])
			    print("nearest distance", distance)
			    print("nearest index", index)
			    print("true label", y_test[0])
			    print("nearest label", y_train[index])
			
			
			def test_time():
			    """
			    测试邻居查找时间
			    :return:
			    """
			    x_train, y_train, x_test, y_test = get_dataset()
			    nnb = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(x_train)
			    start_time = time.time()
			    for sample in x_test:
			        distance, index = nnb.kneighbors([sample])
			    end_time = time.time()
			    print("Spend:{} s".format(end_time-start_time))
			
			
			if __name__ == '__main__':
			    test_accuracy()
			    test_time()
			```
	- 运行结果
		- ![](https://img-blog.csdnimg.cn/20190620203029448.png)
		- 可以看到，zcqk这是效率最高的。
- 算法选择
	- 究竟使用哪种构建和查询更为高效视情况而定，sklearn提供了auto方式，可以自动选择合适的算法。
	- 选择的依据（sklearn也是这个依据）一般根据D(特征维度)和N(数据量)的维度大小决定，具体可以参见sklearn官方文档。
	- 当D很大时Ball tree是最优选择。
- 补充说明
	- 具体代码见我的Github，欢迎star或者fork。