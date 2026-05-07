import sys
import ezdxf
import math

def create_dxf_part_739(out_file):
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()

    doc.layers.add("GEOMETRY", color=1)
    doc.layers.add("DIMENSION", color=3)
    doc.layers.add("ANNOTATION", color=7)

    # Part 739: Connection Plate (from Schnitt B-B)
    # Material: BL 12x217 - 739
    # Overall dimensions: Width 217mm, Height EST 200mm
    
    plate_w = 217
    plate_h = 200 # EST height based on typical connection plate proportions
    thickness = 12
    
    # Outline (LWPOLYLINE)
    points = [
        (0, 0),
        (plate_w, 0),
        (plate_w, plate_h),
        (0, plate_h),
        (0, 0)
    ]
    msp.add_lwpolyline(points, dxfattribs={'layer': 'GEOMETRY'})

    # Holes: 8x M20 (Ø22)
    # 4 holes for L 150x15 (horizontal) - top part
    # 4 holes for L 150x50 (diagonal) - bottom part
    hole_radius_m20 = 22/2
    gauge_line_h = 75 # Gauge for 150mm leg
    
    # Holes for horizontal member (top part of plate)
    # X-coords: EST 50, 100, 150, 200 (spacing 50mm)
    x_coords_h = [50, 100, 150, 200]
    for x in x_coords_h:
        msp.add_circle((x, plate_h - gauge_line_h), hole_radius_m20, dxfattribs={'layer': 'GEOMETRY'})
        
    # Holes for diagonal member (bottom part of plate)
    # X-coords: EST 50, 100, 150, 200 (spacing 50mm)
    x_coords_d = [50, 100, 150, 200]
    for x in x_coords_d:
        msp.add_circle((x, gauge_line_h), hole_radius_m20, dxfattribs={'layer': 'GEOMETRY'})

    # Annotations
    msp.add_mtext("PART 739 - Connection Plate", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, plate_h + 10), 5, 5)
    msp.add_mtext(f"MATERIAL: BL {thickness}x{plate_w} - 739", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, plate_h + 20), 5, 5)
    msp.add_mtext(f"OVERALL: W {plate_w} x H {plate_h} mm (H EST)", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, plate_h + 30), 5, 5)
    msp.add_mtext(f"8x Ø22 (M20)", dxfattribs={'layer': 'ANNOTATION'}).set_location((10, plate_h + 40), 5, 5)

    doc.saveas(out_file)

if __name__ == '__main__':
    create_dxf_part_739(sys.argv[1])