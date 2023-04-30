import cv2
import glob
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
from PIL import Image

def get_descriptor(path = None, queryImage = None) -> 'tuple[tuple, np.ndarray]':
    """
    获取输入图片的描述子

    path: 图片的路径, 应该是完整相对路径
    return: keypoint, descriptor 其中keypoint是关键点元组, descriptor是对应的128维描述子的矩阵
    """
    if type(queryImage) == type(None):
        queryImage = cv2.imread(path)
    
    queryImage = cv2.resize(queryImage, (400, 400))
    gray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoint, descriptor = sift.detectAndCompute(gray, None) # keypoint是关键点构成的列表，descriptor是对应的128维描述子的矩阵

    return keypoint, descriptor

def get_des_list() -> None:
    """
    将所有图片的描述子构成一个list, 并将其保存为des_list.pkl文件
    
    return: None
    """

    des_list = []
    paths = glob.glob('oxford/*.jpg')
    paths.sort()

    i = 0

    for jpg_path in paths:
        _, des = get_descriptor(jpg_path)
        des_list.append(des)
        i += 1
        print(f"{i}/5062 ({100* i/5062:.2f}%) is appended. This picture is {jpg_path}")
    
    with open('des_list.pkl', 'wb') as f:
        pickle.dump(des_list, f)


def get_des_mat() -> None:
    """
    将所有图片的描述子构成一个sum(des)*128维的矩阵, 保存在des_mat.npy文件中, 并返回des_list
    
    return: None
    """

    with open('des_list.pkl', 'rb') as f:
        des_list = pickle.load(f)
    des_mat = np.vstack(des_list)
    np.save('des_mat.npy',des_mat)


def cluster(k) -> None:
    """
    获取所有图像得到的data矩阵, 矩阵的大小是5062*k, 保存在data.npy文件中

    k: 聚类中心数量
    return: None
    """

    des_mat = np.load('des_mat.npy')
    print("the des_mat shape is " + str(des_mat.shape))

    print('cluster start ...')
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1e-2)
    flags = cv2.KMEANS_RANDOM_CENTERS  # 初始化聚类中心的标志
    retval, labels, centers = cv2.kmeans(des_mat, k, None, criteria, 1, flags)
    print('cluster over ...')

    data = []
    with open('des_list.pkl', 'rb') as f:
        des_list = pickle.load(f)

    j=0
    for descriptors in des_list:
        N = descriptors.shape[0]
        labels = np.zeros((N, 1), dtype=int)
        distances = np.zeros((N, k))
        for i in range(N):
            distances[i] = np.sum(np.square(centers - descriptors[i]), axis=1)
            labels[i] = np.argmin(distances[i])
        code, _ = np.histogram(labels, bins=np.arange(k+1))
        j+=1
        data.append(code)
        print(f"Now {j}/5062 ({100* j/5062:.2f}%) code have been added")

    # 将所有图像的特征表示连接起来，形成一个大小为(5062, k)的矩阵
    data = np.vstack(data)
    np.save('data.npy', data)

    with open('centers.pkl', 'wb') as f:
        pickle.dump(centers, f)

def get_feature_vec(des, k) -> list:
    """
    得到一张图片对应的k维特征向量

    des: 一张图片的描述子
    return: list
    """

    with open('model/centers.pkl', 'rb') as f:
        centers = pickle.load(f)

    # print(type(des))
    N = des.shape[0]
    labels = np.zeros((N, 1), dtype=int)
    distances = np.zeros((N, k))
    for i in range(N):
        distances[i] = np.sum(np.square(centers - des[i]), axis=1)
        labels[i] = np.argmin(distances[i])
    code, _ = np.histogram(labels, bins=np.arange(k+1))
    return code

def find_similar_image(query_descriptor, dataset, k) -> list:
    """
    寻找最相似的图片

    name: 图片名称, 方便寻找label, 不含.jpg后缀
    query_descriptor: 待查询图片的描述子
    return: path_list 每一项path是该图片的相对路径(含有jpg后缀)
    """

    data = np.load('model/data.npy') # data是5062xk的矩阵，需要和descriptor进行比较余弦距离

    # print(data.shape)
    
    # 计算查询向量和所有图像特征向量之间的余弦相似度
    similarity_scores = cosine_similarity(get_feature_vec(query_descriptor, k).reshape(1,-1), data)
    # similarity_scores = cosine_similarity(data[2].reshape(1,-1), data)

    # 对相似度得分进行排序，得到所有图像的索引
    indices = similarity_scores.argsort()[0][::-1]

    top_K_indices = indices[0:6]

    # print("top_K_indices is :")
    # print(top_K_indices)
    
    paths = glob.glob(dataset + '/*.jpg')
    paths.sort()

    path_list = []

    i = 0
    for jpg_path in paths:
        if i in top_K_indices:
            path_list.append(jpg_path)
        
        i += 1
        # print(f"{i}/5062 ({100* i/5062:.2f}%) is completed. This picture is {jpg_path}")

    # print(path_list)

    return path_list


def search(name, dataset, k) -> list:
    """
    搜索最相似的5张图片的名称

    dataset: 当前处理的数据集名称
    name: dataset中的图片名称,不含.jpg后缀
    return: 最相似的5张图片的名称组成的list, 不含.jpg后缀
    """
    
    path = dataset + '/' + name + '.jpg'

    _, query_descriptor = get_descriptor(path)

    path_list = find_similar_image(query_descriptor, dataset, k)

    return path_list

def search_similar_image(query_image, dataset_path, k=300) -> list:
    """
    提供给Flask后端的模型接口

    query_image: 需要search的图片,类型:PIL.Image.Image
    dataset_path: 数据集的路径
    k: 模型进行kmeans的簇个数
    """
    # 首先转换为cv2
    query_image = cv2.cvtColor(np.asarray(query_image), cv2.COLOR_RGB2BGR)
    _, query_descriptor = get_descriptor(None, query_image)
    path_list = find_similar_image(query_descriptor, dataset_path, k)
    similar_images = [Image.open(path) for path in path_list]
    return similar_images

if __name__ == '__main__':
    k = 300
    # get_des_list()
    # get_des_mat()
    # cluster(k)
    print(search('all_souls_000051', 'E:/AllDownLoad/images/', k))

    # pass