import struct
import numpy as np


# locs: zyx ordering
def sparse_to_dense_np(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    dense = np.zeros([dimz, dimy, dimx, nf_values], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:,0], locs[:,1], locs[:,2],:] = values
    if nf_values > 1:
        return dense #.reshape([dimz, dimy, dimx, nf_values])
    return dense.reshape([dimz, dimy, dimx])


def load_sdf(file, load_sparse, load_known, load_colors, is_sparse_file=True, color_file=None):
    #assert os.path.isfile(file)
    assert (not load_sparse and not load_known) or (load_sparse != load_known)
    try:
        fin = open(file, 'rb')
        dimx = struct.unpack('Q', fin.read(8))[0]
        dimy = struct.unpack('Q', fin.read(8))[0]
        dimz = struct.unpack('Q', fin.read(8))[0]
        voxelsize = struct.unpack('f', fin.read(4))[0]
        world2grid = struct.unpack('f'*4*4, fin.read(4*4*4))
    except:
        print('failed to read file:', file)
        if load_sparse:
            return None, None, None, None, None
        else:
            return None, None, None, None
    world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
    if is_sparse_file:
        num = struct.unpack('Q', fin.read(8))[0]
        locs = struct.unpack('I'*num*3, fin.read(num*3*4))
        locs = np.asarray(locs, dtype=np.int32).reshape([num, 3])
        locs = np.flip(locs,1).copy() # convert to zyx ordering
        sdf = struct.unpack('f'*num, fin.read(num*4))
        sdf = np.asarray(sdf, dtype=np.float32)
        sdf /= voxelsize
    else:
        raise # unimplemented

    ## load known
    known = None
    num_known = 0
    known_file = None
    if load_colors and color_file is None: # chunk file
        num_known = struct.unpack('Q', fin.read(8))[0]
    if (load_known or num_known > 0) and known_file is None:
        if num_known != dimx * dimy * dimz:
            print('file', file)
            print('dims (%d, %d, %d) -> %d' % (dimx, dimy, dimz, dimx*dimy*dimz))
            print('#known', num_known)
            # input('sdlfkj')
        assert num_known == dimx * dimy * dimz
        known = struct.unpack('B'*num_known, fin.read(num_known))
        if load_known:
            known = np.asarray(known, dtype=np.uint8).reshape([dimz, dimy, dimx])
            mask = np.logical_and(sdf >= -1, sdf <= 1)
            known[locs[:,0][mask], locs[:,1][mask], locs[:,2][mask]] = 1
            mask = sdf > 1
            known[locs[:,0][mask], locs[:,1][mask], locs[:,2][mask]] = 0
        else:
            known = None
    else:
        known = None
    ## load colors
    colors = None
    if load_colors:
        if color_file is not None:
            with open(color_file, 'rb') as cfin:
                cdimx = struct.unpack('Q', cfin.read(8))[0]
                cdimy = struct.unpack('Q', cfin.read(8))[0]
                cdimz = struct.unpack('Q', cfin.read(8))[0]
                assert cdimx == dimx and cdimy == dimy and cdimz == dimz
                if is_sparse_file:
                    num = struct.unpack('Q', cfin.read(8))[0]
                    colors = struct.unpack('B'*num*3, cfin.read(num*3))
                    colors = np.asarray(colors, dtype=np.uint8).reshape(num, 3)
                    #TODO always loads dense
                    colors = sparse_to_dense_np(locs, colors, cdimx, cdimy, cdimz, 0)
                else:
                    colors = struct.unpack('B'*cdimz*cdimy*cdimx*3, cfin.read(cdimz*cdimy*cdimx*3))
                    colors = np.asarray(colors, dtype=np.uint8).reshape([cdimz, cdimy, cdimx, 3])
        else:
            num_color = struct.unpack('Q', fin.read(8))[0]
            assert num_color == dimx * dimy * dimz
            colors = struct.unpack('B'*num_color*3, fin.read(num_color*3))
            colors = np.asarray(colors, dtype=np.uint8).reshape([dimz, dimy, dimx, 3])
    fin.close()
    if load_sparse:
        return [locs, sdf], [dimz, dimy, dimx], world2grid, known, colors
    else:
        sdf = sparse_to_dense_np(locs, sdf[:,np.newaxis], dimx, dimy, dimz, -float('inf'))
        return sdf, world2grid, known, colors

def main():
    """

    notes for the current files:

    - the sdf files are saved sparsely, i.e. the locs and value, the color is saved densely
    - this reader loads all information in zyx order, you can change the order if you need
    - we only need to find the correspondence image id for incomplete input chunks, i.e. the sceneid_roomid__inc__chunkid.sdf, for example: V2XKFyX4ASd_room24__inc__35.sdf

    """

    sdf_file = '/cluster/himring/jhuang/spsg-data/data-geo-color/V2XKFyX4ASd_room24__inc__35.sdf'

    sdf, world2grid, known, colors = load_sdf(sdf_file, load_sparse=False,
                                                        load_known=False,
                                                        load_colors=True, color_file=None)

    print()

if __name__ == '__main__':
    main()