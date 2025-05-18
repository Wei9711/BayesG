import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

path = "NewYork33"  
# 1. Parse junction coordinates from the network file (for fallback when no shape is given)
junctions = {}
for event, elem in ET.iterparse(path + "/newyork33.net.xml", events=("end",)):
    if elem.tag == "junction":
        jid = elem.get("id")
        x, y = elem.get("x"), elem.get("y")
        if jid and x and y:
            junctions[jid] = (float(x), float(y))
        elem.clear()  # free memory

# 2. Parse edges (roads) and get their geometry
edges_geometry = []
for event, elem in ET.iterparse(path + "/newyork33.net.xml", events=("end",)):
    if elem.tag == "edge":
        # Skip internal edges (intersection internals)
        if elem.get("function") == "internal":
            elem.clear()
            continue
        # Determine the polyline for this edge
        coords = []
        shape = elem.get("shape")
        if shape:
            # Use the edge's shape attribute if available
            for point in shape.strip().split():
                x_str, y_str = point.split(',')
                coords.append((float(x_str), float(y_str)))
        else:
            # No edge shape: use the first lane's shape or from/to nodes
            lane = elem.find("lane")
            if lane is not None and lane.get("shape"):
                for point in lane.get("shape").strip().split():
                    x_str, y_str = point.split(',')
                    coords.append((float(x_str), float(y_str)))
            else:
                # Fallback to straight line between from-node and to-node
                from_node = junctions.get(elem.get("from"))
                to_node = junctions.get(elem.get("to"))
                if from_node and to_node:
                    coords = [from_node, to_node]
        if coords:
            edges_geometry.append(coords)
        elem.clear()  # free memory

# 3. Parse neighbor polylines from the additional file
polylines = []
add_tree = ET.parse(path + "/neighbor_graph.add.xml")
for poly in add_tree.getroot().findall("polyline"):
    shape = poly.get("shape")
    color = poly.get("color", "red")   # default to red if not specified
    width = poly.get("width")
    # Convert shape string to list of (x, y) coordinates
    coords = []
    for point in shape.strip().split():
        x_str, y_str = point.split(',')
        coords.append((float(x_str), float(y_str)))
    polylines.append({
        "coords": coords,
        "color": color,
        "width": float(width) if width else 2.0  # default line width 2.0 if not given
    })

# 4. Plot the road network and neighbor polylines using matplotlib
fig, ax = plt.subplots(figsize=(10, 8))
# Plot base road network (edges) in thin gray lines
for coords in edges_geometry:
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    ax.plot(xs, ys, color='gray', linewidth=0.5, zorder=1)  # base roads

# Plot neighbor polylines in a brighter color (e.g., red) with thicker lines
for poly in polylines:
    xs = [p[0] for p in poly["coords"]]
    ys = [p[1] for p in poly["coords"]]
    # Handle color specification (names like "red" or numeric "r,g,b")
    col = poly["color"]
    if ',' in col:
        # e.g., "255,0,0" or "1.0,0.0,0.0" -> convert to RGB tuple
        vals = [float(c) for c in col.split(',')]
        if any(v > 1.0 for v in vals):  # if values seem like 0–255 range
            vals = [v/255.0 for v in vals]
        color_val = tuple(vals[:3])
    else:
        color_val = col  # e.g., "red" or "#FF0000"
    ax.plot(xs, ys, color='#e46d4c', linewidth=poly["width"], zorder=3)  # neighbor line

import json
with open(path + "/tls_positions.json") as f:
    tls_positions = json.load(f)

for tls_id, (x, y) in tls_positions.items():
    ax.plot(x, y, marker='o', markersize=15, color='#1f77b4', zorder=4)


# Set equal aspect ratio for true geometry and add legend for clarity
ax.set_aspect('equal', 'box')
road_line = mlines.Line2D([], [], color='gray', linewidth=1, label='Road network')
poly_line = mlines.Line2D([], [], color='#e46d4c', linewidth=2, label='Edge in Environment Graph')
ax.legend(handles=[road_line, poly_line], loc='upper right')
ax.set_title("NewYork33: SUMO Network with Environment Graph")
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")

plt.tight_layout()
plt.savefig(path + "/network_overlay.png")  # Save the plot to a file
plt.show()  # Display the plot
