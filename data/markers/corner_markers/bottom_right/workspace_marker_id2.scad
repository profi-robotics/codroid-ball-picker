// Standalone workspace marker holder (ID 2) in mm.
// export_part: "combined" (default), "base", or "inlay".
export_part = "combined";
plate_size_mm = 60;
plate_thickness_mm = 5;
marker_size_mm = 40;
marker_recess_mm = 0;
marker_inlay_mm = 0.6;
marker_grid_cells = 6;
label_depth_mm = 0.6;
label_size_mm = 6;
label_offset_mm = 4;
label_margin_mm = 2;
id_label_size_mm = 3.5;
brand_label_size_mm = 2.5;
marker_id = 2;
top_label = "Bottom right";
bottom_label = "Нижній правий";
brand_label = "Profi Robotics Ukraine";

marker_origin = (plate_size_mm - marker_size_mm) / 2;
top_label_y = plate_size_mm - label_margin_mm;
bottom_label_y = label_margin_mm;
marker_cell_size_mm = marker_size_mm / marker_grid_cells;
base_color = [1, 1, 1];
inlay_color = [0, 0, 0];

black_cells = [
  [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
  [1, 0], [1, 1], [1, 2], [1, 5],
  [2, 0], [2, 1], [2, 2], [2, 5],
  [3, 0], [3, 1], [3, 2], [3, 4], [3, 5],
  [4, 0], [4, 3], [4, 5],
  [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5]
];

module marker_recess() {
  translate([marker_origin, marker_origin, plate_thickness_mm - marker_recess_mm])
    cube([marker_size_mm, marker_size_mm, marker_recess_mm + 0.1]);
}

module marker_black_tiles(tile_height_mm, z_offset_mm) {
  for (cell = black_cells) {
    row = cell[0];
    col = cell[1];
    x = marker_origin + col * marker_cell_size_mm;
    y = marker_origin + (marker_grid_cells - 1 - row) * marker_cell_size_mm;
    translate([x, y, z_offset_mm])
      cube([marker_cell_size_mm, marker_cell_size_mm, tile_height_mm]);
  }
}

module label_engrave() {
  translate([plate_size_mm / 2, top_label_y, plate_thickness_mm - label_depth_mm])
    linear_extrude(height = label_depth_mm + 0.1)
      text(
        text = top_label,
        size = label_size_mm,
        font = "Liberation Sans:style=Bold",
        halign = "center",
        valign = "top"
      );
  translate([plate_size_mm / 2, bottom_label_y, plate_thickness_mm - label_depth_mm])
    linear_extrude(height = label_depth_mm + 0.1)
      text(
        text = bottom_label,
        size = label_size_mm,
        font = "Liberation Sans:style=Bold",
        halign = "center",
        valign = "bottom"
      );
  translate([label_margin_mm, plate_size_mm - label_margin_mm, plate_thickness_mm - label_depth_mm])
    linear_extrude(height = label_depth_mm + 0.1)
      text(
        text = str("ID ", marker_id),
        size = id_label_size_mm,
        font = "Liberation Sans:style=Bold",
        halign = "left",
        valign = "top"
      );
  translate([plate_size_mm - label_margin_mm, plate_size_mm - label_margin_mm, plate_thickness_mm - label_depth_mm])
    rotate([0, 0, 90])
      linear_extrude(height = label_depth_mm + 0.1)
        text(
          text = brand_label,
          size = brand_label_size_mm,
          font = "Liberation Sans:style=Bold",
          halign = "right",
          valign = "top"
        );
}

module base_plate() {
  difference() {
    cube([plate_size_mm, plate_size_mm, plate_thickness_mm]);
    if (marker_recess_mm > 0) {
      marker_recess();
    }
    if (marker_inlay_mm > 0) {
      marker_black_tiles(marker_inlay_mm + 0.1, plate_thickness_mm - marker_inlay_mm);
    }
    label_engrave();
  }
}

module inlay_tiles() {
  if (marker_inlay_mm > 0) {
    marker_black_tiles(marker_inlay_mm, plate_thickness_mm - marker_inlay_mm);
  }
}

if (export_part == "base") {
  color(base_color) base_plate();
} else if (export_part == "inlay") {
  color(inlay_color) inlay_tiles();
} else {
  color(base_color) base_plate();
  color(inlay_color) inlay_tiles();
}
