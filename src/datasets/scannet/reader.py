import struct
import numpy as np


class ChunkSDF:
    def __init__(self, dims=[0, 0, 0], res=0.02, chunk_inc_locs=None,
                 ori_world2grid=None, scene_world2grid=None, num_locations=None,
                 nunm_inc_locs=None, locations=None, sdfs=None,
                 Taxisalign=None, Tori=None, chunk_world2grid=None, known=None,
                 colors=None, labels=None, chunk_inc_mask=None):
        self.filename = ""
        self.dimx = dims[0]
        self.dimy = dims[1]
        self.dimz = dims[2]
        self.res = res
        self.ori_world2grid = ori_world2grid
        self.num_locations = num_locations
        self.num_inc_locations = nunm_inc_locs
        self.locations = locations
        self.sdfs = sdfs
        self.Tori = Tori
        self.Taxisalign = Taxisalign
        self.labels = labels
        self.colors = colors
        self.scene_world2grid = scene_world2grid
        self.chunk_world2grid = chunk_world2grid
        self.known = known
        self.chunk_inc_locs = chunk_inc_locs
        self.chunk_inc_mask = chunk_inc_mask


def load_scannet_chunk(file):
    data = ChunkSDF()
    fin = open(file, 'rb')
    data.num_locations = struct.unpack('Q', fin.read(8))[0]
    data.num_inc_locations = struct.unpack('Q', fin.read(8))[0]
    data.dimx = struct.unpack('Q', fin.read(8))[0]
    data.dimy = struct.unpack('Q', fin.read(8))[0]
    data.dimz = struct.unpack('Q', fin.read(8))[0]

    ori_world2grid = struct.unpack('f' * 16, fin.read(16 * 4))
    Tori = struct.unpack('f' * 16, fin.read(16 * 4))
    Taxisalign = struct.unpack('f' * 16, fin.read(16 * 4))
    scene_world2grid = struct.unpack('f' * 16, fin.read(16 * 4))
    chunk_world2grid = struct.unpack('f' * 16, fin.read(16 * 4))

    try:
        location_bytes = fin.read(data.num_locations * 3 * 4)
        locations = struct.unpack('I' * 3 * data.num_locations, location_bytes)

        inc_location_bytes = fin.read(data.num_inc_locations * 3 * 4)
        inc_locations = struct.unpack('I' * 3 * data.num_inc_locations,
                                      inc_location_bytes)

        sdfs_bytes = fin.read(data.num_locations * 4)
        sdfs = struct.unpack('f' * data.num_locations, sdfs_bytes)

        label_bytes = fin.read(data.num_locations * 1)
        labels = struct.unpack('B' * data.num_locations, label_bytes)

        color_bytes = fin.read(data.num_locations * 1 * 3)
        colors = struct.unpack('B' * data.num_locations * 3, color_bytes)

        known_bytes = fin.read(data.num_locations * 1)
        known = struct.unpack('B' * data.num_locations, known_bytes)

    except struct.error as why:
        print(f"Cannot load {file}: {why}")

    fin.close()
    data.ori_world2grid = np.asarray(ori_world2grid, dtype=np.float32).reshape(
        [4, 4])
    data.Tori = np.asarray(Tori, dtype=np.float32).reshape([4, 4])
    data.Taxisalign = np.asarray(Taxisalign, dtype=np.float32).reshape([4, 4])
    data.scene_world2grid = np.asarray(scene_world2grid,
                                       dtype=np.float32).reshape([4, 4])
    data.chunk_world2grid = np.asarray(chunk_world2grid,
                                       dtype=np.float32).reshape([4, 4])
    data.locations = np.asarray(locations, dtype=np.uint32).reshape(
        [data.num_locations, 3], order="C")
    data.chunk_inc_locs = np.asarray(inc_locations, dtype=np.uint32).reshape(
        [data.num_inc_locations, 3], order="C")
    data.sdfs = np.asarray(sdfs, dtype=np.float32)
    data.labels = np.asarray(labels)
    data.known = np.asarray(known)
    data.colors = np.asarray(colors).reshape([data.num_locations, 3])
    # data.labels = map_nyu40_to_20(np.asarray(labels))
    # if load_sparse == False:
    #     data.sdfs = sparse_to_dense_np(data.locations, data.sdfs.reshape(-1, 1), data.dimx, data.dimy, data.dimz, default_val=-float('inf'))
    #     data.labels = sparse_to_dense_np(data.locations, data.labels.reshape(-1, 1), data.dimx, data.dimy, data.dimz, 0)
    #     data.colors = sparse_to_dense_np(data.locations, np.asarray(colors).reshape([data.num_locations, 3]), data.dimx, data.dimy, data.dimz, 0)
    #     chunk_inc_mask = np.zeros((data.dimz,data.dimy,data.dimx))
    #     chunk_inc_mask[data.chunk_inc_locs[:, 0], data.chunk_inc_locs[:, 1], data.chunk_inc_locs[:, 2]] = 1
    #     data.chunk_inc_mask = chunk_inc_mask
    return data


class ScannetSDF:
    def __init__(self, dims=[0, 0, 0], res=0, world2grid=None, num_locations=None, locations=None, sdfs=None,Taxisalign = None, Tori = None,colors = None,labels = None):
        self.filename = ""
        self.dimx = dims[0]
        self.dimy = dims[1]
        self.dimz = dims[2]
        self.res = res
        self.world2grid = world2grid
        self.num_locations = num_locations
        self.locations = locations
        self.sdfs = sdfs
        self.Tori = Tori
        self.Taxisalign = Taxisalign
        self.labels = labels
        self.colors = colors

# sem: zyx ordering
def sparse_to_dense_sem(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    dense = np.zeros([dimx, dimy, dimz, nf_values], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:,0], locs[:,1], locs[:,2],:] = values.reshape(-1,1)
    if nf_values > 1:
        return dense #.reshape([dimz, dimy, dimx, nf_values])
    return dense.reshape([dimx, dimy, dimz])

# locs: zyx ordering
def sparse_to_dense_np(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    dense = np.zeros([dimz, dimy, dimx, nf_values], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:,0], locs[:,1], locs[:,2],:] = values
    if nf_values > 1:
        return dense #.reshape([dimz, dimy, dimx, nf_values])
    return dense.reshape([dimz, dimy, dimx])

# zyx ordering
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
    if load_colors and color_file is None: # chunk file
        num_known = struct.unpack('Q', fin.read(8))[0]
    if load_known or num_known > 0:
        if num_known != dimx * dimy * dimz:
            print('file', file)
            print('dims (%d, %d, %d) -> %d' % (dimx, dimy, dimz, dimx*dimy*dimz))
            print('#known', num_known)
            raw_input('sdlfkj')
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


# zyx ordering
def load_sem_sparse2dense(file):
    dimxx = 64
    dimyy = 64
    dimzz = 128
    fin = open(file, 'rb')
    dimx = struct.unpack('i', fin.read(4))[0]
    dimy = struct.unpack('i', fin.read(4))[0]
    locs = struct.unpack('h' * dimx * dimy, fin.read(dimx * dimy * 2))
    labels = struct.unpack('h' * dimx, fin.read(dimx * 2))
    fin.close()

    locs = np.asarray(locs).reshape([dimx, dimy])
    labels = np.asarray(labels)

    sem_map = sparse_to_dense_sem(locs, labels, dimxx,dimyy,dimzz, 0)
    sem_label = np.transpose(sem_map, (2, 1, 0))
    return sem_label

def map_nyu40_to_20(target):
    nyu40_to_20_map = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, '11':11, '12':12, '13':0,'14':13, '15':0,'16':14, '17':0,
    '18':0,'19':0,'20':0,'21':0,'22':0,'23':0,'24':15,'25':0,'26':0,'27':0,'28':16,'29':0,'30':0,'31':0,'32':0,'33':17,'34':18,'35':0,'36':19,'37':0,'38':0,'39':20,'40':0}

    target = target.astype(str).reshape(-1)
    nyu20_target = np.array(list(map(nyu40_to_20_map.get, target)))

    return nyu20_target


def load_scannet_sdf(file,load_sparse):
    data = ScannetSDF()
    fin = open(file, 'rb')
    data.num_locations = struct.unpack('Q', fin.read(8))[0]
    data.dimx = struct.unpack('Q', fin.read(8))[0]
    data.dimy = struct.unpack('Q', fin.read(8))[0]
    data.dimz = struct.unpack('Q', fin.read(8))[0]
    # data.res = struct.unpack('f', fin.read(4))[0]

    world2grid = struct.unpack('f' * 16, fin.read(16 * 4))
    Tori = struct.unpack('f' * 16, fin.read(16 * 4))
    Taxisalign = struct.unpack('f' * 16, fin.read(16 * 4))

    try:
        location_bytes = fin.read(data.num_locations * 3 * 4)
        locations = struct.unpack('I' * 3 * data.num_locations, location_bytes)

        sdfs_bytes = fin.read(data.num_locations * 4)
        sdfs = struct.unpack('f' * data.num_locations, sdfs_bytes)

        label_bytes = fin.read(data.num_locations * 1)
        labels = struct.unpack('B' * data.num_locations, label_bytes)

        color_bytes = fin.read(data.num_locations * 1 * 3)
        colors = struct.unpack('B' * data.num_locations* 3, color_bytes)

    except struct.error:
        print("Cannot load", file)

    fin.close()
    data.world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
    data.Tori = np.asarray(Tori, dtype=np.float32).reshape([4, 4])
    data.Taxisalign = np.asarray(Taxisalign, dtype=np.float32).reshape([4, 4])
    data.locations = np.asarray(locations, dtype=np.uint32).reshape([data.num_locations, 3], order="C")
    data.locations =  np.flip(data.locations,1).copy()
    data.sdfs = np.asarray(sdfs, dtype=np.float32)

    if load_sparse==False:
        data.sdfs=sparse_to_dense_np(
            data.locations,
            data.sdfs.reshape(-1,1),
            data.dimx, data.dimy, data.dimz,
            default_val=-float('inf'))

    ## if needed:
    data.labels = map_nyu40_to_20(np.asarray(labels))
    data.labels = data.labels.reshape(-1, 1)
    data.colors = np.asarray(colors).reshape([data.num_locations, 3])
    # data.labels = sparse_to_dense_np(
    #     data.locations,
    #     data.labels.reshape(-1, 1),
    #     data.dimx, data.dimy, data.dimz, 0)
    # data.colors = sparse_to_dense_np(
    #     data.locations,
    #     np.asarray(colors).reshape([data.num_locations, 3]),
    #     data.dimx, data.dimy, data.dimz, 0)
    return data # zyx ordering


def load_sdf_128(file, load_sparse, load_known, load_colors, is_sparse_file=True, color_file=None):
    ''' this version for generate the new chunks  with 128 cube'''
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
    # if load_colors and color_file is None:  # chunk file
    #     num_known = struct.unpack('Q', fin.read(8))[0]
    if load_known :
        num_known = struct.unpack('Q', fin.read(8))[0]

        if num_known != dimx * dimy * dimz:
            print('file', file)
            print('dims (%d, %d, %d) -> %d' % (dimx, dimy, dimz, dimx * dimy * dimz))
            print('#known', num_known)
            print('num_known != dimx * dimy * dimz')
        assert num_known == dimx * dimy * dimz
        known = struct.unpack('B' * num_known, fin.read(num_known))
        known = np.asarray(known, dtype=np.uint8).reshape([dimz, dimy, dimx])
        mask = np.logical_and(sdf >= -1, sdf <= 1)
        known[locs[:, 0][mask], locs[:, 1][mask], locs[:, 2][mask]] = 1
        mask = sdf > 1
        known[locs[:, 0][mask], locs[:, 1][mask], locs[:, 2][mask]] = 0
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
            if num_color == dimx * dimy * dimz:
                colors = struct.unpack('B'*num_color*3, fin.read(num_color*3))
                colors = np.asarray(colors, dtype=np.uint8).reshape([dimz, dimy, dimx, 3])
            else:
                colors = struct.unpack('B' * num_color * 3, fin.read(num_color * 3))
                colors = np.asarray(colors, dtype=np.uint8).reshape(num, 3)
                colors = sparse_to_dense_np(locs, colors, dimx, dimy, dimz, 0)
    fin.close()
    if load_sparse:
        return [locs, sdf], [dimz, dimy, dimx], world2grid, known, colors
    else:
        sdf = sparse_to_dense_np(locs, sdf[:,np.newaxis], dimx, dimy, dimz, -float('inf'))
        return sdf, known, colors


def load_sem_sparse2dense_128(file,dimxx,dimyy,dimzz,map40to20=True):
    fin = open(file, 'rb')
    dimx = struct.unpack('i', fin.read(4))[0]
    dimy = struct.unpack('i', fin.read(4))[0]
    world2grid = struct.unpack('f' * 16, fin.read(16 * 4))
    locs = struct.unpack('h' * dimx * dimy, fin.read(dimx * dimy * 2))
    labels = struct.unpack('h' * dimx, fin.read(dimx * 2))
    fin.close()

    locs = np.asarray(locs).reshape([dimx, dimy])
    labels = np.asarray(labels)
    world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])

    if map40to20==True:
        labels = map_nyu40_to_20(labels)
    sem_map = sparse_to_dense_sem(locs, labels, dimxx,dimyy,dimzz, 0)
    return sem_map,world2grid


def load_inc_mask(file_name):
    '''load by zyx ordering'''
    fin = open(file_name, 'rb')
    num_locations = struct.unpack('Q', fin.read(8))[0]
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    location_bytes = fin.read(num_locations * 3 * 4)
    locations = struct.unpack('I' * 3 * num_locations, location_bytes) # zyx ordering
    locations = np.asarray(locations, dtype=np.uint32).reshape([num_locations, 3], order="C")
    fin.close()

    return locations

if __name__ == '__main__':
    sdf_folder = '/cluster/himring/jhuang/spsg-data/data-geo-color/'
    train_ls = '/rhome/jhuang/3drecon/spsg-2dsemantic/filelists/chunk64/sdf_frame_check_train_ls.txt'
    val_ls = '/rhome/jhuang/3drecon/spsg-2dsemantic/filelists/chunk64/sdf_frame_check_train_ls.txt'

    with open(train_ls) as f:
        input_ls = f.readlines()

    for file in input_ls:
        input_sdf_path = sdf_folder + file.replace('\n','')
        target_sdf_path = input_sdf_path.replace('__inc__','__cmp__')
        target_sem_path = target_sdf_path.replace('.sdf', '-sparse-sem.npy').replace('data-geo-color', 'sem3d-label-13cat-chunks-64').replace('himring', 'angmar')

        ## load input/incomplete chunk like this
        '''if you want to load dense/sparse sdf, set load_sparse to False/True
           input files dont have known(visibility) mask, so set load_known to False here'''
        input, _, _, _, input_colors = load_sdf(input_sdf_path, load_sparse=True, load_known=False,load_colors=True, color_file=None)

        # load target/complete chunk like this:
        '''if you want to load dense/sparse sdf, set load_sparse to False/True
            target files have known(visibility) mask, so set load_known to True here'''
        sdf, world2grid, known, colors = load_sdf(target_sdf_path, load_sparse=False,load_known=True ,load_colors=True, color_file=None)

        ## load sem label like this:
        '''14 classes, range from 0 to 13, 0 denotes unlabeled, set weight=0 for class 0 when training semantic segmentation '''
        sem_label = load_sem_sparse2dense(target_sem_path)


        '''scannet dataset, full-scene data, nyu classes, set load_sparse=True if you want to load sparse data, the final world2grid should be like scannet_data.world2grid @ scannet_data.Tori @ scannet_data.Taxisalign'''
        scannet_data = load_scannet_sdf('path to .npy file',load_sparse=True)
        final_world2grid = scannet_data.world2grid @ scannet_data.Tori @ scannet_data.Taxisalign

        '''matterport3d, for chunksize=128'''
        ### to load complete data:
        sdf, known, colors = load_sdf_128('path to incomplete sdf',  load_sparse=False, load_known=True, load_colors=True, is_sparse_file=True, color_file=None)

        ### to load incomplete data:
        sdf_input, known_input, colors_input = load_sdf_128('path to complete sdf', load_sparse=False, load_known=False, load_colors=True, is_sparse_file=True, color_file=None)

        ## to load semantic label, and also the world2grid matrix:
        sem_label, world2grid = load_sem_sparse2dense_128('path to semantic file', 128, 128, 128, map40to20=True)


        ### to load scannet incomplete mask
        '''the mask files save the location of the incomplete sdf, note that the location is saved in zyx ordering'''
        load_inc_locs = load_inc_mask('path to mask file')
