# CV-object-retrieval

CV课设，选题一。

数据集比较大，所以没有打包数据集。

## 运行方法

1. [下载数据集](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz)，然后在运行app.py时添加-path参数。

    _如果需要单独运行search.py，还需要单独更改该文件中的路径_

2. 安装项目所需的包。

    ```
    $ pip install -r requirements.txt -i https://pypi.douban.com/simple
    ```

3. cd到app目录下。

    ```
    $ cd ./app
    ```

4. 部署项目到本地，端口号可以在app.py文件中更改。

    ```
    $ python ./app.py -path your_dataset_path
    ```