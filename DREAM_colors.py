import numpy as np
import pandas as pd

blue = [{"name":"Alice Blue","hex":"DFEBF9","rgb":[223,235,249],"cmyk":[10,6,0,2],"hsb":[212,10,98],"hsl":[212,68,93],"lab":[93,-1,-8]},{"name":"Jordy Blue","hex":"9FC3ED","rgb":[159,195,237],"cmyk":[33,18,0,7],"hsb":[212,33,93],"hsl":[212,68,78],"lab":[78,-2,-25]},{"name":"Blue Gray","hex":"5F9BE1","rgb":[95,155,225],"cmyk":[58,31,0,12],"hsb":[212,58,88],"hsl":[212,68,63],"lab":[63,1,-41]},{"name":"Lapis Lazuli","hex":"305E94","rgb":[48,94,148],"cmyk":[68,36,0,42],"hsb":[212,68,58],"hsl":[212,51,38],"lab":[39,3,-34]}]
green = [{"name":"Tiffany Blue","hex":"6ACDC1","rgb":[106,205,193],"cmyk":[48,0,6,20],"hsb":[173,48,80],"hsl":[173,50,61],"lab":[76,-32,-3]},{"name":"Persian green","hex":"07AB98","rgb":[7,171,152],"cmyk":[96,0,11,33],"hsb":[173,96,67],"hsl":[173,92,35],"lab":[63,-41,-1]},{"name":"Skobeloff","hex":"16817A","rgb":[22,129,122],"cmyk":[83,0,5,49],"hsb":[176,83,51],"hsl":[176,71,30],"lab":[49,-30,-4]},{"name":"Midnight green","hex":"024249","rgb":[2,66,73],"cmyk":[97,10,0,71],"hsb":[186,97,29],"hsl":[186,95,15],"lab":[25,-15,-10]}]
orange = [{"name":"Pale Dogwood","hex":"FFD9CA","rgb":[255,217,202],"cmyk":[0,15,21,0],"hsb":[17,21,100],"hsl":[17,100,90],"lab":[89,11,12]},{"name":"Coral","hex":"F79071","rgb":[247,144,113],"cmyk":[0,42,54,3],"hsb":[14,54,97],"hsl":[14,89,71],"lab":[70,36,33]},{"name":"Jasper","hex":"BB573B","rgb":[187,87,59],"cmyk":[0,53,68,27],"hsb":[13,68,73],"hsl":[13,52,48],"lab":[49,38,35]},{"name":"Persian green","hex":"07AB98","rgb":[7,171,152],"cmyk":[96,0,11,33],"hsb":[173,96,67],"hsl":[173,92,35],"lab":[63,-41,-1]}]

blue_df = pd.DataFrame(blue)
green_df = pd.DataFrame(green)
orange_df = pd.DataFrame(orange)

colors = pd.concat([green_df, orange_df, blue_df], ignore_index=True)
colors.rgb = colors.rgb.apply(lambda x: np.array(x)/255)

def get_random_colors(n):
    return colors.rgb.sample(n).tolist()