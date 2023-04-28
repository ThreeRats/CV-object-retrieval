import cv2
import numpy
import glob
from sklearn.cluster import KMeans
import joblib
import numpy as np
import json

def get_descriptor(path) -> 'tuple[tuple, numpy.ndarray]':
    """
    获取输入图片的描述子

    path: 图片的路径, 应该是完整相对路径
    return: keypoint, descriptor 其中keypoint是关键点元组, descriptor是对应的128维描述子的矩阵
    """

    queryImage = cv2.imread(path, 0)
    sift = cv2.SIFT_create()
    keypoint, descriptor = sift.detectAndCompute(queryImage, None) # keypoint是关键点构成的列表，descriptor是对应的128维描述子的矩阵

    return keypoint, descriptor

def reshape_des(des) -> np.ndarray:
    """
    使用描述子的平均值用于聚类

    des: 图片的描述子, 是128列矩阵
    return: np.array([均值])
    """

    reshape_des = des.reshape(-1,1)
    reshape_feature_mean = np.array([np.mean(reshape_des)])

    return reshape_feature_mean

def cluster() -> None:
    """
    获取所有图片的聚类模型，并保存

    return: None
    """

    X = []
    paths = glob.glob('oxford/*.jpg')
    paths.sort()

    i = 0
    for jpg_path in paths:
        _, des = get_descriptor(jpg_path)
        X.append(reshape_des(des).tolist())
        i += 1
        print(f"{i}/5062 ({100* i/5062:.2f}%) is appended. This picture is {jpg_path}")

    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=100) # 原始图片有5062张, 聚类中心设置为100个
    kmeans.fit(X)

    joblib.dump(kmeans, 'kmeans.pkl')

def preprocess_dataset() -> None:
    """
    重新根据kmeans结果将图片划分到不同的文件夹中

    return: None
    """

    kmeans = joblib.load('kmeans.pkl')

    paths = glob.glob('oxford/*.jpg')
    paths.sort()
    names = [path.split('\\')[-1][:-4] for path in paths]

    name_label_dict = dict(zip(names, kmeans.labels_.tolist()))

    with open('name_label.json', 'w') as f:
        json.dump(name_label_dict, f)

def find_similar_image(name, query_descriptor, dataset) -> list:
    """
    KD树寻找最相似的图片

    name: 图片名称, 方便寻找label, 不含.jpg后缀
    query_descriptor: 待查询图片的描述子
    return: mean_list 每一项是一个字典, 字典包含两个属性path和mean, path是该图片的相对路径, mean是前50个关键点的描述子的distance均值
    """

    FLANN_INDEX_KDTREE = 0 # 使用KD-Tree算法进行最近邻搜索

    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)

    flann = cv2.FlannBasedMatcher(indexParams, searchParams) # FLANN匹配器

    with open('name_label.json') as json_file:
        name_label_dict = json.load(json_file)
    
    query_label = name_label_dict[name]

    
    paths = glob.glob(dataset + '/*.jpg')
    paths.sort()

    mean_list = []

    i = 0

    for jpg_path in paths:
        label = name_label_dict[jpg_path.split("\\")[-1][:-4]]
        if query_label == label:
            _, judge_descriptor = get_descriptor(jpg_path)
            matches = flann.match(query_descriptor, judge_descriptor)
            matches = sorted(matches, key=lambda x: x.distance)
            mean_list.append( {'path':jpg_path, 
                            'mean':numpy.mean([x.distance for x in matches[:50] if x.distance != 0])} )
        
        # print_info(jpg_path, matches)
        i += 1
        print(f"{i}/5062 ({100* i/5062:.2f}%) is completed. This picture is {jpg_path}")

    
    mean_list = sorted(mean_list, key=lambda x: x['mean'])

    return mean_list

# def print_info(jpg_path, matches) -> None:
#     """
#     寻找过程中打印日志

#     dataset: 当前处理的数据集名称
#     jpg_path: 当前处理图片的相对路径
#     matches: matches数组, 已经经过排序
#     return: None
#     """

#     print(f'-----Now is computing {jpg_path}-----')
#     print(f'the sorted matches is {matches}')



def search(name, dataset) -> list:
    """
    搜索最相似的5张图片的名称

    dataset: 当前处理的数据集名称
    name: dataset中的图片名称,不含.jpg后缀
    return: 最相似的5张图片的名称组成的list, 不含.jpg后缀
    """
    
    path = dataset + '/' + name + '.jpg'

    _, query_descriptor = get_descriptor(path)

    mean_list = find_similar_image(name, query_descriptor, dataset)

    name_list = []

    for img in mean_list[1:6]: # 第一张应该是原图
        img_path = img['path']
        name = img_path.split('\\')[1]
        name = name.split('.')[0]
        name_list.append(name)

    return name_list


if __name__ == '__main__':
    print(search('all_souls_000000', 'oxford'))
    # cluster()
    # preprocess_dataset()
    # pass