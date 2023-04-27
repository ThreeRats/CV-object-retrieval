import cv2
import numpy
import glob

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

def find_similar_image(query_descriptor) -> list:
    """
    KD树寻找最相似的图片

    query_descriptor: 待查询图片的描述子
    return: mean_list 每一项是一个字典, 字典包含两个属性path和mean, path是该图片的相对路径, mean是前50个关键点的描述子的distance均值
    """
    
    FLANN_INDEX_KDTREE = 0 # 使用KD-Tree算法进行最近邻搜索

    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)

    flann = cv2.FlannBasedMatcher(indexParams, searchParams) # FLANN匹配器

    
    paths = glob.glob('bupt/*.jpg')
    paths.sort()

    mean_list = []

    for jpg_path in paths:
        _, judge_descriptor = get_descriptor(jpg_path)
        matches = flann.match(query_descriptor, judge_descriptor)
        matches = sorted(matches, key=lambda x: x.distance)
        mean_list.append( {'path':jpg_path, 
                           'mean':numpy.mean([x.distance for x in matches[:50]])} )
        
        # print_info(jpg_path, matches)
    
    mean_list = sorted(mean_list, key=lambda x: x['mean'])

    return mean_list

# def print_info(jpg_path, matches) -> None:
#     """
#     寻找过程中打印日志

#     jpg_path: 当前处理图片的相对路径
#     matches: matches数组, 已经经过排序
#     return: None
#     """

#     print(f'-----Now is computing {jpg_path}-----')
#     print(f'the sorted matches is {matches}')



def search(name) -> list:
    """
    搜索最相似的5张图片的名称
    
    name: dataset中的图片名称,不含.jpg后缀
    return: 最相似的5张图片的名称组成的list, 不含.jpg后缀
    """
    
    path = 'bupt/' + name + '.jpg'

    _, query_descriptor = get_descriptor(path)

    mean_list = find_similar_image(query_descriptor)

    name_list = []

    for img in mean_list:
        img_path = img['path']
        name = img_path.split('\\')[1]
        name = name.split('.')[0]
        name_list.append(name)

    return name_list


if __name__ == '__main__':
    print(search('bupt2'))