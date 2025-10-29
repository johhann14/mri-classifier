import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def plot_split_proportions(df):
    counts = (df['split']
              .value_counts(normalize=True)
              .mul(100)                      
              .rename('percent')            
              .reset_index())                
    
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=counts, x='split', y='percent')
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.1f}%',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center',
                    va='center',
                    xytext=(0, 5), 
                    textcoords='offset points')

    plt.title('Proportion Train vs Test (%)')
    plt.xlabel('Split')
    plt.ylabel('%')
    plt.ylim(0, counts['percent'].max() * 1.1)
    plt.show()

def plot_class_distribution(df, split_name):
    df_split = df[df['split'] == split_name]['label']
    counts = df_split.value_counts()
    total = counts.sum()
    

    counts_df = counts.reset_index()
    counts_df.columns = ['Class Label', 'Count']
    
    counts_df['Percentage'] = (counts_df['Count'] / total) * 100

    fig, ax = plt.subplots(figsize=(8, 6)) 
    
    ax = sns.barplot(
        x='Class Label', 
        y='Count', 
        data=counts_df, 
        ax=ax,
        hue='Class Label',
        legend=False,
        palette='viridis'
    )
    
    for i, p in enumerate(ax.patches):
        count = counts_df['Count'].iloc[i]
        percentage = counts_df['Percentage'].iloc[i]
        text_label = f'{count} ({percentage:.1f}%)' 
        
        ax.text(
            p.get_x() + p.get_width() / 2., 
            p.get_height(),
            s=text_label, 
            ha='center', 
            va='bottom',
            fontsize=10
        )
    
    plt.title(f"{split_name.capitalize()} : Class Counts and Proportions")
    plt.xlabel('Class Label')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_sample_images(df, n_samples=5):
    samples = df.sample(n_samples, random_state=14)
    plt.figure(figsize=(22,22))
    for i in range(n_samples):
        row = samples.iloc[i]
        img = cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        plt.subplot(n_samples//6 + 1, 6, i+1)
        plt.imshow(img, cmap='gray')
        height, width = img.shape
        plt.title(f"Label: {row['label']}\nDim: {width}x{height}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_metrics(df):
    columns = ['mean', 'std', 'skewness', 'kurtosis', 'entropy', 'contrast', 'energy', 'asm', 'homogeneity', 'dissimilarity', 'correlation']
    n = len(columns)
    fig, ax = plt.subplots(n//4 +1 ,4, figsize=(25,20))
    ax = ax.ravel()

    for i, col in enumerate(columns):
        sns.histplot(data=df, x=col, kde=True, ax=ax[i], hue='label')
        ax[i].set_title(col, fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_PCA(df):
    columns = ['mean', 'std', 'skewness', 'kurtosis', 'entropy', 'contrast', 'energy', 'asm', 'homogeneity', 'dissimilarity', 'correlation']
    scarler = StandardScaler()
    X = df[columns].values
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=df.index)

    pca_df['label'] = df['label']

    # 3. Affichage (Plotting)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='PC1', 
        y='PC2', 
        data=pca_df, 
        hue='label',           
        palette='tab10', 
        s=50                 
    )
    
    plt.title('2D PCA Projection of Data by Class', fontsize=16)
    plt.show()


    
