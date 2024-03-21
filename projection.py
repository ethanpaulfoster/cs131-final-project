def project_2d_to_3d(point, f, c_x, c_y, Z):
    #given point [x, y], projection matrix params, and the known Z coordinate, return the 3d coordinates of the point
    u, v = point
    x_hat = (u-c_x)/f
    y_hat = (v-c_y)/f
    
    x = x_hat * Z
    y = y_hat * Z
    z = Z
    
    return [x, y, z]

def diver_height_and_distance(board_coordinates, diver_coordinates, f, c_x, c_y, board_z_distance):
    #params:
        #board_coordinates
        #diver_coordinates
        #f, c_x, c_y
        #board_z_distance
    #returns: 
        #height above the board, distance from the board
        
    board_3d = project_2d_to_3d(board_coordinates, f, c_x, c_y, board_z_distance)
    diver_3d = project_2d_to_3d(diver_coordinates, f, c_x, c_y, board_z_distance)
    
    height = board_3d[1] - diver_3d[1]
    distance = board_3d[0] - diver_3d[0]
    
    return height, distance