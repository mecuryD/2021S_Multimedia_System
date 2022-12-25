import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

'''
Basis function (COMPLETE)
'''

def basis_uv(u,v) :
    basis = np.zeros((8,8))
    # Set Cu, Cv
    Cu = 1 if u else 1/np.sqrt(2)
    Cv = 1 if v else 1/np.sqrt(2)
        
    # Compute basis at [u][v]
    for i in range(0,8) :
      for j in range(0,8) :
          basis[i][j] = ((Cu*Cv) / 4)*np.cos(((2*i + 1)*u*np.pi)/16)*np.cos(((2*j + 1)*v*np.pi) /16)
    return basis

def make_basis() :
    basis = []
    
    for u in range(0,8):
        for v in range(0,8):
            basis.append(basis_uv(u,v))
    
    return basis
            
'''
Discrete Cosine Transform
'''

def dct_block(block, basis) :
    F =[]
    
    for i in range(len(basis)) :
        F.append(np.sum(np.multiply(block,basis[i])))
    
    return np.array(F).reshape((8,8))

def dct(img,basis) :
    img_dct = np.zeros(img.shape)
    
    for i in range(int(img.shape[0]/8)) :
        for j in range(int(img.shape[1]/8)) :
            x = i * 8
            y = j * 8
            img_dct[x:x+8, y:y+8] = dct_block(img[x:x+8, y:y+8],basis)
    
    return img_dct

def idct_block(block, basis, size=8) :
    f = np.zeros((8, 8))
    
    if(size!=8) :
        for i in range(block.shape[0]) :
            for j in range(block.shape[1]) :
                if (i>(size-1))|(j>(size-1)) :
                    block[i][j] = 0
    block = block.reshape((1, 64))
    
    for i in range(len(basis)) :
        f += np.multiply(block[0, i],basis[i])
        
    return f

def idct(img,basis,size) :
    img_idct = np.zeros(img.shape)
    
    for i in range(int(img.shape[0]/8)) :
        for j in range(int(img.shape[1]/8)) :
            x = i * 8
            y = j * 8
            img_idct[x:x+8, y:y+8] = idct_block(img[x:x+8, y:y+8],basis,size)
    
    return img_idct


'''
Load Images (COMPLETE)
'''
img = cv.imread('lena.jpg', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (120,120))

plt.imshow(img,cmap='gray', vmin = 0, vmax = 255)
plt.axis("off")
plt.title("Original")
plt.show()

'''
Discrete Cosine Transform
'''
basis = make_basis()
img_dct = dct(img, basis)
img_idct1 = idct(img_dct,basis,8)
img_idct2 = idct(img_dct,basis,4)
img_idct3 = idct(img_dct,basis,2)

# Plot 8x8 dct-idct
plt.imshow(img_idct1,cmap='gray', vmin = 0, vmax = 255)
plt.axis("off")
plt.title("Transformed : 8x8")
plt.show()

# Plot 4x4 dct-idct
plt.imshow(img_idct2,cmap='gray', vmin = 0, vmax = 255)
plt.axis("off")
plt.title("Transformed : 4x4")
plt.show()

# Plot 2x2 dct-idct
plt.imshow(img_idct3,cmap='gray', vmin = 0, vmax = 255)
plt.axis("off")
plt.title("Transformed : 2x2")
plt.show()


