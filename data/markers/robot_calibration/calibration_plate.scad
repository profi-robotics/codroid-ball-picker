// Robot calibration plate for Codroid (units in mm).
// export_part: "combined" (default), "base", "inlay", or "groove".
// Marker centers follow SVG-style coordinates: origin at top-left, +x right, +y down.
export_part = "combined";
plate_width_mm = 300.0;
plate_height_mm = 280.0;
plate_thickness_mm = 5;
marker_size_mm = 20.0;
marker_recess_mm = 0;
marker_grid_cells = 6;
marker_inlay_mm = 0.6;
marker_cell_size_mm = marker_size_mm / marker_grid_cells;
label_size_mm = 8;
label_depth_mm = 0.6;
label_offset_mm = 10.0;
margin_bottom_mm = 20.0;
groove_width_mm = 20.0;
groove_height_mm = 33.0;
groove_depth_mm = 2.6;
groove_inlay_mm = 0.6;
footer_offset_mm = 35.0;
footer_text = "Estun Codroid Coordinate System calibration board by Profi Robotics Ukraine";
footer_text_size_mm = 4;
footer_text_depth_mm = 0.6;
base_color = [1, 1, 1];
inlay_color = [0, 0, 0];

marker_centers_mm = [
  [20, 20.0, 20.0],
  [21, 150.0, 20.0],
  [22, 280.0, 20.0],
  [23, 20.0, 140.0],
  [24, 280.0, 140.0],
  [25, 20.0, 260.0],
  [26, 150.0, 260.0],
  [27, 280.0, 260.0]
];

calibration_points_mm = [
  [1, 70.0, 70.0],
  [2, 230.0, 70.0],
  [3, 230.0, 210.0]
];

black_cells_id20 = [
  [0, 0],
  [0, 1],
  [0, 2],
  [0, 3],
  [0, 4],
  [0, 5],
  [1, 0],
  [1, 2],
  [1, 3],
  [1, 4],
  [1, 5],
  [2, 0],
  [2, 1],
  [2, 4],
  [2, 5],
  [3, 0],
  [3, 2],
  [3, 3],
  [3, 4],
  [3, 5],
  [4, 0],
  [4, 2],
  [4, 5],
  [5, 0],
  [5, 1],
  [5, 2],
  [5, 3],
  [5, 4],
  [5, 5]
];
black_cells_id21 = [
  [0, 0],
  [0, 1],
  [0, 2],
  [0, 3],
  [0, 4],
  [0, 5],
  [1, 0],
  [1, 2],
  [1, 5],
  [2, 0],
  [2, 1],
  [2, 2],
  [2, 3],
  [2, 4],
  [2, 5],
  [3, 0],
  [3, 1],
  [3, 2],
  [3, 4],
  [3, 5],
  [4, 0],
  [4, 2],
  [4, 5],
  [5, 0],
  [5, 1],
  [5, 2],
  [5, 3],
  [5, 4],
  [5, 5]
];
black_cells_id22 = [
  [0, 0],
  [0, 1],
  [0, 2],
  [0, 3],
  [0, 4],
  [0, 5],
  [1, 0],
  [1, 3],
  [1, 4],
  [1, 5],
  [2, 0],
  [2, 3],
  [2, 4],
  [2, 5],
  [3, 0],
  [3, 3],
  [3, 5],
  [4, 0],
  [4, 1],
  [4, 3],
  [4, 5],
  [5, 0],
  [5, 1],
  [5, 2],
  [5, 3],
  [5, 4],
  [5, 5]
];
black_cells_id23 = [
  [0, 0],
  [0, 1],
  [0, 2],
  [0, 3],
  [0, 4],
  [0, 5],
  [1, 0],
  [1, 3],
  [1, 5],
  [2, 0],
  [2, 3],
  [2, 5],
  [3, 0],
  [3, 2],
  [3, 3],
  [3, 4],
  [3, 5],
  [4, 0],
  [4, 1],
  [4, 2],
  [4, 4],
  [4, 5],
  [5, 0],
  [5, 1],
  [5, 2],
  [5, 3],
  [5, 4],
  [5, 5]
];
black_cells_id24 = [
  [0, 0],
  [0, 1],
  [0, 2],
  [0, 3],
  [0, 4],
  [0, 5],
  [1, 0],
  [1, 5],
  [2, 0],
  [2, 4],
  [2, 5],
  [3, 0],
  [3, 1],
  [3, 3],
  [3, 4],
  [3, 5],
  [4, 0],
  [4, 1],
  [4, 5],
  [5, 0],
  [5, 1],
  [5, 2],
  [5, 3],
  [5, 4],
  [5, 5]
];
black_cells_id25 = [
  [0, 0],
  [0, 1],
  [0, 2],
  [0, 3],
  [0, 4],
  [0, 5],
  [1, 0],
  [1, 2],
  [1, 3],
  [1, 5],
  [2, 0],
  [2, 1],
  [2, 3],
  [2, 4],
  [2, 5],
  [3, 0],
  [3, 1],
  [3, 5],
  [4, 0],
  [4, 1],
  [4, 2],
  [4, 3],
  [4, 5],
  [5, 0],
  [5, 1],
  [5, 2],
  [5, 3],
  [5, 4],
  [5, 5]
];
black_cells_id26 = [
  [0, 0],
  [0, 1],
  [0, 2],
  [0, 3],
  [0, 4],
  [0, 5],
  [1, 0],
  [1, 2],
  [1, 4],
  [1, 5],
  [2, 0],
  [2, 3],
  [2, 4],
  [2, 5],
  [3, 0],
  [3, 4],
  [3, 5],
  [4, 0],
  [4, 1],
  [4, 3],
  [4, 4],
  [4, 5],
  [5, 0],
  [5, 1],
  [5, 2],
  [5, 3],
  [5, 4],
  [5, 5]
];
black_cells_id27 = [
  [0, 0],
  [0, 1],
  [0, 2],
  [0, 3],
  [0, 4],
  [0, 5],
  [1, 0],
  [1, 2],
  [1, 4],
  [1, 5],
  [2, 0],
  [2, 1],
  [2, 3],
  [2, 5],
  [3, 0],
  [3, 1],
  [3, 3],
  [3, 5],
  [4, 0],
  [4, 1],
  [4, 3],
  [4, 4],
  [4, 5],
  [5, 0],
  [5, 1],
  [5, 2],
  [5, 3],
  [5, 4],
  [5, 5]
];

function black_cells_for(marker_id) =
  marker_id == 20 ? black_cells_id20 :
  marker_id == 21 ? black_cells_id21 :
  marker_id == 22 ? black_cells_id22 :
  marker_id == 23 ? black_cells_id23 :
  marker_id == 24 ? black_cells_id24 :
  marker_id == 25 ? black_cells_id25 :
  marker_id == 26 ? black_cells_id26 :
  marker_id == 27 ? black_cells_id27 : [];

function to_scad(point_xy) = [point_xy[0], plate_height_mm - point_xy[1]];

module marker_recess(center_xy) {
  let(scad_xy = to_scad(center_xy))
    translate([
      scad_xy[0] - marker_size_mm / 2,
      scad_xy[1] - marker_size_mm / 2,
      plate_thickness_mm - marker_recess_mm
    ])
      cube([marker_size_mm, marker_size_mm, marker_recess_mm + 0.1]);
}

module marker_black_tiles(center_xy, marker_id, tile_height_mm, z_offset_mm) {
  marker_origin_x = center_xy[0] - marker_size_mm / 2;
  marker_origin_y = center_xy[1] - marker_size_mm / 2;
  for (cell = black_cells_for(marker_id)) {
    row = cell[0];
    col = cell[1];
    tile_x = marker_origin_x + col * marker_cell_size_mm;
    tile_y = marker_origin_y + row * marker_cell_size_mm;
    translate([tile_x, plate_height_mm - (tile_y + marker_cell_size_mm), z_offset_mm])
      cube([marker_cell_size_mm, marker_cell_size_mm, tile_height_mm]);
  }
}

function label_y_for(center_xy) =
  (center_xy[1] + groove_height_mm / 2 + label_offset_mm > plate_height_mm - margin_bottom_mm) ?
  (center_xy[1] - groove_height_mm / 2 - label_offset_mm) :
  (center_xy[1] + groove_height_mm / 2 + label_offset_mm);

module sequence_label(center_xy, label_text) {
  label_xy = [center_xy[0], label_y_for(center_xy)];
  scad_xy = to_scad(label_xy);
  translate([scad_xy[0], scad_xy[1], plate_thickness_mm - label_depth_mm])
    linear_extrude(height=label_depth_mm + 0.1)
      text(
        text = str(label_text),
        size = label_size_mm,
        font = "Liberation Sans:style=Bold",
        halign = "center",
        valign = "center"
      );
}

module calibration_groove(center_xy) {
  scad_xy = to_scad(center_xy);
  translate([
    scad_xy[0] - groove_width_mm / 2,
    scad_xy[1] - groove_height_mm / 2,
    plate_thickness_mm - groove_depth_mm
  ])
    cube([groove_width_mm, groove_height_mm, groove_depth_mm + 0.1]);
}

module groove_inlay(center_xy) {
  if (groove_inlay_mm > 0) {
    scad_xy = to_scad(center_xy);
    translate([
      scad_xy[0] - groove_width_mm / 2,
      scad_xy[1] - groove_height_mm / 2,
      plate_thickness_mm - groove_depth_mm
    ])
      cube([groove_width_mm, groove_height_mm, groove_inlay_mm]);
  }
}

module footer_label() {
  footer_xy = [plate_width_mm / 2, plate_height_mm - footer_offset_mm];
  scad_xy = to_scad(footer_xy);
  translate([scad_xy[0], scad_xy[1], plate_thickness_mm - footer_text_depth_mm])
    linear_extrude(height=footer_text_depth_mm + 0.1)
      text(
        text = footer_text,
        size = footer_text_size_mm,
        font = "Liberation Sans:style=Bold",
        halign = "center",
        valign = "center"
      );
}

module base_plate() {
  difference() {
    cube([plate_width_mm, plate_height_mm, plate_thickness_mm]);
    for (marker = marker_centers_mm) {
      if (marker_recess_mm > 0) {
        marker_recess([marker[1], marker[2]]);
      }
      if (marker_inlay_mm > 0) {
        marker_black_tiles([marker[1], marker[2]], marker[0], marker_inlay_mm + 0.1, plate_thickness_mm - marker_inlay_mm);
      }
    }
    for (point = calibration_points_mm) {
      calibration_groove([point[1], point[2]]);
      sequence_label([point[1], point[2]], point[0]);
    }
    footer_label();
  }
}

module inlay_tiles() {
  if (marker_inlay_mm > 0) {
    for (marker = marker_centers_mm) {
      marker_black_tiles([marker[1], marker[2]], marker[0], marker_inlay_mm, plate_thickness_mm - marker_inlay_mm);
    }
  }
}

module groove_inlays() {
  for (point = calibration_points_mm) {
    groove_inlay([point[1], point[2]]);
  }
}

if (export_part == "base") {
  color(base_color) base_plate();
} else if (export_part == "inlay") {
  color(inlay_color) inlay_tiles();
} else if (export_part == "groove") {
  color(base_color) groove_inlays();
} else {
  color(base_color) base_plate();
  color(inlay_color) inlay_tiles();
  color(base_color) groove_inlays();
}
