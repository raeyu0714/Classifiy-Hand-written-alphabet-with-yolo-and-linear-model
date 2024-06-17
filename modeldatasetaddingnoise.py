from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
import pandas as pd
data = pd.read_csv('final.csv',header=None)
result_df = pd.DataFrame()
for i in range(26):
    for j in range(800):
        dd = data[data[0]==i].iloc[j]
        x = dd[1:].values
        x = x.reshape((28, 28))
        '''
        plt.imshow(x, cmap='binary')
        plt.show()
        '''
        img_gray = dd
        noise = np.random.normal(0, 50, img_gray.shape) 
        img_noised = img_gray + noise
        img_noised = np.clip(img_noised, 0, 255).astype(np.uint8)
        y = img_noised[1:].values
        df = pd.DataFrame(y)
        dft=df.transpose()
        dft.loc[0,0]=i
        #dft.iloc[0][0]=i
        print(dft.iloc[0][0])
        result_df = pd.concat([result_df,dft])
        '''
        y = y.reshape((28, 28))
        plt.imshow(y, cmap='binary')
        plt.show()
        '''
result_df.to_csv('finaln.csv', index = False)
