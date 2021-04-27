import numpy as np
import skimage.io as io

class TexTransformer:
    def __init__(self, path_to_texmap):
        self.texmap = np.load(path_to_texmap)

    def GetGlobalUV(self, I, UV):
        I = I.reshape(-1)
        U = UV[:,:,0].reshape(-1)
        V = UV[:,:,1].reshape(-1)
        UV_global = -1*np.ones((I.shape[0], 2))
        for ci in range(1, 25):
            inds = (I==ci).nonzero()
            u = U[inds]
            v = V[inds]
            u_gl = self.texmap[ci-1][u, v, 0]
            v_gl = self.texmap[ci - 1][u, v, 1]
            v_gl = np.ones_like(v_gl)-2*v_gl
            u_gl = 2*u_gl - np.ones_like(u_gl)
            UV_global[inds, 0] = u_gl
            UV_global[inds, 1] = v_gl

        out = UV_global.reshape(UV.shape[0], UV.shape[1], 2)
        out[out==-1.] = np.nan

        return out

    def ShowSurrealTexture(self, UV_global, texpath):
        tex_img = io.imread(texpath)

        im_flat = np.zeros((UV_global.shape[0]*UV_global.shape[1], 3), dtype=np.uint8)
        uv_flat = UV_global.reshape(-1, 2)
        uv_flat = (tex_img.shape[0]-1)/2.0*(uv_flat + np.ones_like(uv_flat))
        uv_int = np.floor(uv_flat + 0.5*np.ones_like(uv_flat)).astype(int)
        tex_inds = np.nonzero(uv_flat[:, 0]>0)[0]
        uv_int = uv_int[tex_inds]
        im_flat[tex_inds, :] = tex_img[uv_int[:, 1], uv_int[:,0], :]
        out = im_flat.reshape(UV_global.shape[0], UV_global.shape[1], 3)
        
        return out
