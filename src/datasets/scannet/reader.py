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
