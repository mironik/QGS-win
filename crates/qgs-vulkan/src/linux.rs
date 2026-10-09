use std::any::Any;
use std::ffi::CStr;
use std::fs::File;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ash::vk;
use qgs_core::{
    BackendBufferAllocation, BackendImageAllocation, BackendResource, BackendResourceExport,
    BackendSync, BackendSyncExport, DeviceDiscovery, DeviceDiscoveryError, ResourceBackend,
    ResourceError, SyncBackend, SyncError, VideoCapabilityDiscovery, VideoCapabilityDiscoveryError,
};
use qgs_protocol::{
    ApiVersion, BackendApi, BufferDesc, BufferUsageFlags, ComputeCapabilities, CreateSyncRequest,
    DeviceCapabilities, DeviceClass, DeviceDesc, DeviceId, ExportResourceRequest,
    ExportSyncRequest, ExportedResourceMetadata, ExportedSyncMetadata, ExternalHandleType,
    ExternalSharing, ImageDesc, ImageUsageFlags, InteropCapabilities, MemoryCapabilities,
    MemoryHeapDesc, PixelFormat, ResourceKind, SelectedMemoryProperties, SyncExportHandleType,
    SyncKind, VideoCapabilities, MAX_DEVICE_COUNT, MAX_DEVICE_NAME_LEN, MAX_MEMORY_HEAP_COUNT,
    MAX_MEMORY_TYPE_COUNT, MAX_VIDEO_SURFACE_HEIGHT, MAX_VIDEO_SURFACE_WIDTH,
};

mod haswell_video_diagnostic;

pub use haswell_video_diagnostic::{
    diagnose_haswell_video_import, DiagnosticDrmLayer, DiagnosticDrmObject, DiagnosticDrmPlane,
    HaswellVideoDiagnosticInput, HaswellVideoDiagnosticReport,
};

const SHARED_VALIDATION_MARKER: &[u8] = b"QGS-M1S6";
const COMPUTE_LOCAL_SIZE_X: u32 = 64;
const COMPUTE_INCREMENT_SHADER: [u32; 147] = [
    119734787, 65536, 0, 22, 0, 131089, 1, 720906, 1599492179, 1599227979, 1919906931, 1600481121,
    1717990754, 1935635045, 1634889588, 1667196263, 1936941420, 0, 196622, 0, 1, 393231, 5, 1,
    1852399981, 0, 9, 393232, 1, 17, 64, 1, 1, 196611, 2, 450, 262215, 9, 11, 28, 262215, 10, 6, 4,
    327752, 11, 0, 35, 0, 196679, 11, 2, 262215, 13, 34, 0, 262215, 13, 33, 0, 131091, 2, 196641,
    3, 2, 262165, 4, 32, 0, 262187, 4, 5, 1, 262187, 4, 6, 0, 262167, 7, 4, 3, 262176, 8, 1, 7,
    262203, 8, 9, 1, 196637, 10, 4, 196638, 11, 10, 262176, 12, 12, 11, 262203, 12, 13, 12, 262176,
    14, 1, 4, 262176, 15, 12, 4, 327734, 2, 1, 0, 3, 131320, 16, 327745, 14, 17, 9, 6, 262205, 4,
    18, 17, 393281, 15, 19, 13, 6, 18, 262205, 4, 20, 19, 327808, 4, 21, 20, 5, 196670, 19, 21,
    65789, 65592,
];
const IMAGE_INVERT_SHADER: [u32; 208] = [
    119734787, 65536, 0, 35, 0, 131089, 1, 196622, 0, 1, 393231, 5, 3, 1852399981, 0, 11, 393232,
    3, 17, 8, 8, 1, 196611, 2, 450, 262149, 3, 1852399981, 0, 524293, 11, 1197436007, 1633841004,
    1986939244, 1952539503, 1231974249, 68, 196613, 14, 6778217, 262215, 11, 11, 28, 262215, 14,
    34, 0, 262215, 14, 33, 0, 131091, 1, 196641, 2, 1, 262165, 4, 32, 0, 262165, 5, 32, 1, 196630,
    6, 32, 262167, 7, 4, 3, 262167, 8, 5, 2, 262167, 9, 6, 4, 262176, 10, 1, 7, 262203, 10, 11, 1,
    589849, 12, 6, 1, 0, 0, 0, 2, 4, 262176, 13, 0, 12, 262203, 13, 14, 0, 262187, 4, 15, 0,
    262187, 4, 16, 1, 262187, 6, 17, 1065353216, 327734, 1, 3, 0, 2, 131320, 18, 262205, 7, 19, 11,
    327761, 4, 20, 19, 0, 327761, 4, 21, 19, 1, 262268, 5, 22, 20, 262268, 5, 23, 21, 327760, 8,
    24, 22, 23, 262205, 12, 25, 14, 327778, 9, 26, 25, 24, 327761, 6, 27, 26, 0, 327761, 6, 28, 26,
    1, 327761, 6, 29, 26, 2, 327761, 6, 30, 26, 3, 327816, 6, 31, 17, 27, 327816, 6, 32, 17, 28,
    327816, 6, 33, 17, 29, 458832, 9, 34, 31, 32, 33, 30, 262243, 25, 24, 34, 65789, 65592,
];
const YUV422P10_LOCAL_SIZE_X: u32 = 8;
const YUV422P10_LOCAL_SIZE_Y: u32 = 8;
const YUV422P10_OUTPUT_COMPONENTS: usize = 4;
const YUV422P10_OUTPUT_BYTES_PER_COMPONENT: usize = 2;
const YUV422P10_TO_RGBA_U16_SHADER: [u32; 740] = [
    119734787, 66816, 851979, 205, 0, 131089, 1, 393227, 1, 1280527431, 1685353262, 808793134, 0,
    196622, 0, 1, 720911, 5, 4, 1852399981, 0, 23, 36, 76, 88, 98, 160, 393232, 4, 17, 8, 8, 1,
    262215, 23, 11, 28, 196679, 34, 2, 327752, 34, 0, 35, 0, 327752, 34, 1, 35, 4, 327752, 34, 2,
    35, 8, 327752, 34, 3, 35, 12, 262215, 73, 6, 4, 196679, 74, 2, 262216, 74, 0, 24, 327752, 74,
    0, 35, 0, 196679, 76, 24, 262215, 76, 33, 0, 262215, 76, 34, 0, 262215, 85, 6, 4, 196679, 86,
    2, 262216, 86, 0, 24, 327752, 86, 0, 35, 0, 196679, 88, 24, 262215, 88, 33, 1, 262215, 88, 34,
    0, 262215, 95, 6, 4, 196679, 96, 2, 262216, 96, 0, 24, 327752, 96, 0, 35, 0, 196679, 98, 24,
    262215, 98, 33, 2, 262215, 98, 34, 0, 262215, 157, 6, 16, 196679, 158, 2, 262216, 158, 0, 25,
    327752, 158, 0, 35, 0, 196679, 160, 25, 262215, 160, 33, 3, 262215, 160, 34, 0, 262215, 180,
    11, 25, 131091, 2, 196641, 3, 2, 196630, 6, 32, 262187, 6, 13, 0, 262187, 6, 14, 1065353216,
    262165, 18, 32, 0, 262167, 21, 18, 3, 262176, 22, 1, 21, 262203, 22, 23, 1, 262187, 18, 24, 0,
    262176, 25, 1, 18, 262187, 18, 29, 1, 131092, 32, 393246, 34, 18, 18, 18, 18, 262176, 35, 9,
    34, 262203, 35, 36, 9, 262165, 37, 32, 1, 262187, 37, 38, 0, 262176, 39, 9, 18, 262187, 37, 47,
    1, 262187, 37, 57, 2, 262187, 37, 65, 3, 196637, 73, 18, 196638, 74, 73, 262176, 75, 12, 74,
    262203, 75, 76, 12, 262176, 78, 12, 18, 262187, 18, 81, 1023, 196637, 85, 18, 196638, 86, 85,
    262176, 87, 12, 86, 262203, 87, 88, 12, 196637, 95, 18, 196638, 96, 95, 262176, 97, 12, 96,
    262203, 97, 98, 12, 262187, 6, 106, 1115684864, 262187, 6, 113, 1140850688, 262167, 156, 18, 4,
    196637, 157, 156, 196638, 158, 157, 262176, 159, 12, 158, 262203, 159, 160, 12, 262187, 6, 163,
    1199570688, 262187, 18, 175, 65535, 262176, 177, 12, 156, 262187, 18, 179, 8, 393260, 21, 180,
    179, 179, 29, 262187, 6, 199, 982884389, 262187, 6, 201, 988176106, 262187, 6, 202, 962279677,
    262187, 6, 203, 973665712, 262187, 6, 204, 990361937, 327734, 2, 4, 0, 3, 131320, 5, 196855,
    181, 0, 196859, 24, 182, 131320, 182, 327745, 25, 26, 23, 24, 262205, 18, 27, 26, 327745, 25,
    30, 23, 29, 262205, 18, 31, 30, 327745, 39, 40, 36, 38, 262205, 18, 41, 40, 327854, 32, 42, 27,
    41, 262312, 32, 43, 42, 196855, 45, 0, 262394, 43, 44, 45, 131320, 44, 327745, 39, 48, 36, 47,
    262205, 18, 49, 48, 327854, 32, 50, 31, 49, 131321, 45, 131320, 45, 458997, 32, 51, 42, 182,
    50, 44, 196855, 53, 0, 262394, 51, 52, 53, 131320, 52, 131321, 181, 131320, 53, 327745, 39, 58,
    36, 57, 262205, 18, 59, 58, 327812, 18, 60, 31, 59, 327808, 18, 62, 60, 27, 327745, 39, 66, 36,
    65, 262205, 18, 67, 66, 327812, 18, 68, 31, 67, 327874, 18, 70, 27, 47, 327808, 18, 71, 68, 70,
    393281, 78, 79, 76, 38, 62, 262205, 18, 80, 79, 327879, 18, 82, 80, 81, 262256, 6, 83, 82,
    393281, 78, 90, 88, 38, 71, 262205, 18, 91, 90, 327879, 18, 92, 91, 81, 262256, 6, 93, 92,
    393281, 78, 100, 98, 38, 71, 262205, 18, 101, 100, 327879, 18, 102, 101, 81, 262256, 6, 103,
    102, 327811, 6, 107, 83, 106, 458764, 6, 108, 1, 40, 13, 107, 327813, 6, 110, 108, 199, 327811,
    6, 114, 93, 113, 327811, 6, 119, 103, 113, 327813, 6, 125, 119, 201, 327809, 6, 126, 110, 125,
    524300, 6, 190, 1, 43, 126, 13, 14, 327813, 6, 133, 114, 202, 327811, 6, 134, 110, 133, 327813,
    6, 137, 119, 203, 327811, 6, 138, 134, 137, 524300, 6, 194, 1, 43, 138, 13, 14, 327813, 6, 145,
    114, 204, 327809, 6, 146, 110, 145, 524300, 6, 198, 1, 43, 146, 13, 14, 327812, 18, 153, 31,
    41, 327808, 18, 155, 153, 27, 327813, 6, 164, 190, 163, 393228, 6, 165, 1, 1, 164, 262253, 18,
    166, 165, 327813, 6, 168, 194, 163, 393228, 6, 169, 1, 1, 168, 262253, 18, 170, 169, 327813, 6,
    172, 198, 163, 393228, 6, 173, 1, 1, 172, 262253, 18, 174, 173, 458832, 156, 176, 166, 170,
    174, 175, 393281, 177, 178, 160, 38, 155, 196670, 178, 176, 131321, 181, 131320, 181, 65789,
    65592,
];
const NV12_TO_RGBA_U16_SHADER: [u32; 1342] = [
    119734787, 66304, 851979, 218, 0, 131089, 1, 393227, 1, 1280527431, 1685353262, 808793134, 0,
    196622, 0, 1, 393231, 5, 4, 1852399981, 0, 84, 393232, 4, 17, 8, 8, 1, 196611, 2, 450, 655364,
    1197427783, 1279741775, 1885560645, 1953718128, 1600482425, 1701734764, 1919509599, 1769235301,
    25974, 524292, 1197427783, 1279741775, 1852399429, 1685417059, 1768185701, 1952671090, 6649449,
    262149, 4, 1852399981, 0, 393221, 10, 1684107116, 1954112095, 679042917, 3879285, 327685, 9,
    1702132066, 1684957535, 30821, 458757, 13, 1684107116, 1954112095, 1987403621, 993097000, 0,
    327685, 12, 1702132066, 1684957535, 30821, 327685, 19, 1835101283, 674312304, 3879270, 196613,
    18, 118, 327685, 23, 1969188724, 1713911345, 15153, 196613, 22, 118, 262149, 25, 1685221239, 0,
    262149, 27, 1634488409, 25966, 327686, 27, 0, 1870094201, 7562354, 262149, 29, 1819303801,
    6647393, 262149, 49, 1685221239, 0, 262149, 51, 1817204309, 6647393, 393222, 51, 0, 2002744949,
    1935962735, 0, 327685, 53, 1885304437, 1701732716, 0, 262149, 72, 1634886000, 109, 196613, 81,
    120, 524293, 84, 1197436007, 1633841004, 1986939244, 1952539503, 1231974249, 68, 196613, 89,
    7827314, 262149, 95, 1634885968, 29549, 327686, 95, 0, 1952737655, 104, 327686, 95, 1,
    1734960488, 29800, 458758, 95, 2, 1953718137, 1701079410, 1954112095, 29541, 458758, 95, 3,
    1935636085, 1684632180, 2036490085, 7562612, 196613, 97, 25456, 196613, 114, 30329, 262149,
    122, 1634886000, 109, 327685, 124, 1767863925, 2019910766, 0, 196613, 135, 7758435, 262149,
    136, 1634886000, 109, 196613, 139, 7762531, 262149, 142, 1634886000, 109, 196613, 144, 31097,
    196613, 152, 25187, 196613, 159, 29283, 196613, 164, 114, 262149, 170, 1634886000, 109, 196613,
    172, 103, 262149, 182, 1634886000, 109, 196613, 184, 98, 262149, 190, 1634886000, 109, 196613,
    192, 7890025, 196613, 201, 7632207, 327686, 201, 0, 1633838962, 0, 327685, 203, 1601467759,
    1633838962, 0, 262215, 26, 6, 4, 196679, 27, 2, 262216, 27, 0, 24, 327752, 27, 0, 35, 0,
    196679, 29, 24, 262215, 29, 33, 0, 262215, 29, 34, 0, 262215, 50, 6, 4, 196679, 51, 2, 262216,
    51, 0, 24, 327752, 51, 0, 35, 0, 196679, 53, 24, 262215, 53, 33, 1, 262215, 53, 34, 0, 262215,
    84, 11, 28, 196679, 95, 2, 327752, 95, 0, 35, 0, 327752, 95, 1, 35, 4, 327752, 95, 2, 35, 8,
    327752, 95, 3, 35, 12, 262215, 200, 6, 4, 196679, 201, 2, 262216, 201, 0, 25, 327752, 201, 0,
    35, 0, 196679, 203, 25, 262215, 203, 33, 2, 262215, 203, 34, 0, 262215, 217, 11, 25, 131091, 2,
    196641, 3, 2, 262165, 6, 32, 0, 262176, 7, 7, 6, 262177, 8, 6, 7, 196630, 15, 32, 262176, 16,
    7, 15, 262177, 17, 15, 16, 262177, 21, 6, 16, 196637, 26, 6, 196638, 27, 26, 262176, 28, 12,
    27, 262203, 28, 29, 12, 262165, 30, 32, 1, 262187, 30, 31, 0, 262187, 6, 33, 2, 262176, 35, 12,
    6, 262187, 6, 40, 3, 262187, 6, 42, 8, 262187, 6, 45, 255, 196637, 50, 6, 196638, 51, 50,
    262176, 52, 12, 51, 262203, 52, 53, 12, 262187, 15, 67, 0, 262187, 15, 68, 1065353216, 262187,
    15, 75, 1199570688, 262167, 82, 6, 3, 262176, 83, 1, 82, 262203, 83, 84, 1, 262187, 6, 85, 0,
    262176, 86, 1, 6, 262187, 6, 90, 1, 131092, 93, 393246, 95, 6, 6, 6, 6, 262176, 96, 9, 95,
    262203, 96, 97, 9, 262176, 98, 9, 6, 262187, 30, 106, 1, 262187, 30, 116, 2, 262187, 30, 127,
    3, 262187, 6, 132, 4294967294, 262187, 15, 147, 1098907648, 262187, 15, 150, 1130037248,
    262187, 15, 155, 1124073472, 262187, 15, 157, 1130364928, 262187, 15, 166, 1070174988, 262187,
    15, 174, 1044369885, 262187, 15, 178, 1055895027, 262187, 15, 186, 1072530509, 196637, 200, 6,
    196638, 201, 200, 262176, 202, 12, 201, 262203, 202, 203, 12, 262187, 6, 207, 16, 262187, 6,
    214, 4294901760, 393260, 82, 217, 42, 42, 90, 327734, 2, 4, 0, 3, 131320, 5, 262203, 7, 81, 7,
    262203, 7, 89, 7, 262203, 7, 114, 7, 262203, 7, 122, 7, 262203, 7, 124, 7, 262203, 7, 135, 7,
    262203, 7, 136, 7, 262203, 7, 139, 7, 262203, 7, 142, 7, 262203, 16, 144, 7, 262203, 16, 152,
    7, 262203, 16, 159, 7, 262203, 7, 164, 7, 262203, 16, 170, 7, 262203, 7, 172, 7, 262203, 16,
    182, 7, 262203, 7, 184, 7, 262203, 16, 190, 7, 262203, 7, 192, 7, 327745, 86, 87, 84, 85,
    262205, 6, 88, 87, 196670, 81, 88, 327745, 86, 91, 84, 90, 262205, 6, 92, 91, 196670, 89, 92,
    262205, 6, 94, 81, 327745, 98, 99, 97, 31, 262205, 6, 100, 99, 327854, 93, 101, 94, 100,
    262312, 93, 102, 101, 196855, 104, 0, 262394, 102, 103, 104, 131320, 103, 262205, 6, 105, 89,
    327745, 98, 107, 97, 106, 262205, 6, 108, 107, 327854, 93, 109, 105, 108, 131321, 104, 131320,
    104, 458997, 93, 110, 101, 5, 109, 103, 196855, 112, 0, 262394, 110, 111, 112, 131320, 111,
    65789, 131320, 112, 262205, 6, 115, 89, 327745, 98, 117, 97, 116, 262205, 6, 118, 117, 327812,
    6, 119, 115, 118, 262205, 6, 120, 81, 327808, 6, 121, 119, 120, 196670, 122, 121, 327737, 6,
    123, 10, 122, 196670, 114, 123, 262205, 6, 125, 89, 327814, 6, 126, 125, 33, 327745, 98, 128,
    97, 127, 262205, 6, 129, 128, 327812, 6, 130, 126, 129, 262205, 6, 131, 81, 327879, 6, 133,
    131, 132, 327808, 6, 134, 130, 133, 196670, 124, 134, 262205, 6, 137, 124, 196670, 136, 137,
    327737, 6, 138, 13, 136, 196670, 135, 138, 262205, 6, 140, 124, 327808, 6, 141, 140, 90,
    196670, 142, 141, 327737, 6, 143, 13, 142, 196670, 139, 143, 262205, 6, 145, 114, 262256, 15,
    146, 145, 327811, 15, 148, 146, 147, 458764, 15, 149, 1, 40, 148, 67, 327816, 15, 151, 149,
    150, 196670, 144, 151, 262205, 6, 153, 135, 262256, 15, 154, 153, 327811, 15, 156, 154, 155,
    327816, 15, 158, 156, 157, 196670, 152, 158, 262205, 6, 160, 139, 262256, 15, 161, 160, 327811,
    15, 162, 161, 155, 327816, 15, 163, 162, 157, 196670, 159, 163, 262205, 15, 165, 144, 262205,
    15, 167, 159, 327813, 15, 168, 166, 167, 327809, 15, 169, 165, 168, 196670, 170, 169, 327737,
    6, 171, 23, 170, 196670, 164, 171, 262205, 15, 173, 144, 262205, 15, 175, 152, 327813, 15, 176,
    174, 175, 327811, 15, 177, 173, 176, 262205, 15, 179, 159, 327813, 15, 180, 178, 179, 327811,
    15, 181, 177, 180, 196670, 182, 181, 327737, 6, 183, 23, 182, 196670, 172, 183, 262205, 15,
    185, 144, 262205, 15, 187, 152, 327813, 15, 188, 186, 187, 327809, 15, 189, 185, 188, 196670,
    190, 189, 327737, 6, 191, 23, 190, 196670, 184, 191, 262205, 6, 193, 89, 327745, 98, 194, 97,
    31, 262205, 6, 195, 194, 327812, 6, 196, 193, 195, 262205, 6, 197, 81, 327808, 6, 198, 196,
    197, 327812, 6, 199, 198, 33, 196670, 192, 199, 262205, 6, 204, 192, 262205, 6, 205, 164,
    262205, 6, 206, 172, 327876, 6, 208, 206, 207, 327877, 6, 209, 205, 208, 393281, 35, 210, 203,
    31, 204, 196670, 210, 209, 262205, 6, 211, 192, 327808, 6, 212, 211, 90, 262205, 6, 213, 184,
    327877, 6, 215, 213, 214, 393281, 35, 216, 203, 31, 212, 196670, 216, 215, 65789, 65592,
    327734, 6, 10, 0, 8, 196663, 7, 9, 131320, 11, 262203, 7, 25, 7, 262205, 6, 32, 9, 327874, 6,
    34, 32, 33, 393281, 35, 36, 29, 31, 34, 262205, 6, 37, 36, 196670, 25, 37, 262205, 6, 38, 25,
    262205, 6, 39, 9, 327879, 6, 41, 39, 40, 327812, 6, 43, 41, 42, 327874, 6, 44, 38, 43, 327879,
    6, 46, 44, 45, 131326, 46, 65592, 327734, 6, 13, 0, 8, 196663, 7, 12, 131320, 14, 262203, 7,
    49, 7, 262205, 6, 54, 12, 327874, 6, 55, 54, 33, 393281, 35, 56, 53, 31, 55, 262205, 6, 57, 56,
    196670, 49, 57, 262205, 6, 58, 49, 262205, 6, 59, 12, 327879, 6, 60, 59, 40, 327812, 6, 61, 60,
    42, 327874, 6, 62, 58, 61, 327879, 6, 63, 62, 45, 131326, 63, 65592, 327734, 15, 19, 0, 17,
    196663, 16, 18, 131320, 20, 262205, 15, 66, 18, 524300, 15, 69, 1, 43, 66, 67, 68, 131326, 69,
    65592, 327734, 6, 23, 0, 21, 196663, 16, 22, 131320, 24, 262203, 16, 72, 7, 262205, 15, 73, 22,
    196670, 72, 73, 327737, 15, 74, 19, 72, 327813, 15, 76, 74, 75, 393228, 15, 77, 1, 1, 76,
    262253, 6, 78, 77, 131326, 78, 65592,
];

type BackendResult<T> = Result<T, BackendError>;

#[derive(Clone, Copy, Debug)]
enum BackendError {
    Vk(vk::Result),
    NoMemoryType,
    InvalidInput,
}

impl From<vk::Result> for BackendError {
    fn from(value: vk::Result) -> Self {
        Self::Vk(value)
    }
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vk(result) => write!(f, "vulkan error {result:?}"),
            Self::NoMemoryType => write!(f, "no compatible Vulkan memory type"),
            Self::InvalidInput => write!(f, "invalid Vulkan backend input"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum YcbcrConversion {
    Rec709Limited,
}

#[derive(Clone, Copy, Debug)]
pub struct Yuv422P10Plane<'a> {
    pub width_samples: u32,
    pub height: u32,
    pub stride_bytes: usize,
    pub data: &'a [u8],
}

#[derive(Clone, Copy, Debug)]
pub struct Yuv422P10Upload<'a> {
    pub device_id: DeviceId,
    pub width: u32,
    pub height: u32,
    pub y: Yuv422P10Plane<'a>,
    pub cb: Yuv422P10Plane<'a>,
    pub cr: Yuv422P10Plane<'a>,
    pub conversion: YcbcrConversion,
}

#[derive(Clone, Copy, Debug)]
pub struct Nv12Plane<'a> {
    pub width_bytes: u32,
    pub height: u32,
    pub stride_bytes: usize,
    pub data: &'a [u8],
}

#[derive(Clone, Copy, Debug)]
pub struct Nv12Upload<'a> {
    pub device_id: DeviceId,
    pub coded_width: u32,
    pub coded_height: u32,
    pub visible_width: u32,
    pub visible_height: u32,
    pub y: Nv12Plane<'a>,
    pub uv: Nv12Plane<'a>,
    pub conversion: YcbcrConversion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Yuv422P10GpuOutput {
    pub width: u32,
    pub height: u32,
    pub rgba_u16: Vec<u16>,
    pub checksum: u64,
    pub cpu_surface_bytes: usize,
    pub staging_bytes: usize,
    pub gpu_plane_bytes: usize,
    pub output_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Nv12GpuOutput {
    pub width: u32,
    pub height: u32,
    pub rgba_u16: Vec<u16>,
    pub checksum: u64,
    pub cpu_surface_bytes: usize,
    pub staging_bytes: usize,
    pub gpu_plane_bytes: usize,
    pub output_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameToken(u64);

impl FrameToken {
    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameIdentity {
    pub presentation_position: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuFrameProcessorConfig {
    pub device_id: DeviceId,
    pub width: u32,
    pub height: u32,
    pub slot_count: usize,
    pub conversion: YcbcrConversion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Nv12FrameProcessorConfig {
    pub device_id: DeviceId,
    pub coded_width: u32,
    pub coded_height: u32,
    pub visible_width: u32,
    pub visible_height: u32,
    pub slot_count: usize,
    pub conversion: YcbcrConversion,
    pub validation_readback: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuFrameProcessorCounters {
    pub pipeline_creations: u64,
    pub shader_module_creations: u64,
    pub gpu_plane_allocations: u64,
    pub staging_allocations: u64,
    pub output_allocations: u64,
    pub readback_allocations: u64,
    pub command_buffer_count: u64,
    pub frame_submissions: u64,
    pub slot_reuses: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Nv12FrameProcessorDiagnostics {
    pub slot_acquire_ns: u128,
    pub compact_y_copy_ns: u128,
    pub compact_uv_copy_ns: u128,
    pub staging_write_y_ns: u128,
    pub staging_write_uv_ns: u128,
    pub command_pool_reset_ns: u128,
    pub fence_reset_ns: u128,
    pub command_begin_ns: u128,
    pub command_record_ns: u128,
    pub command_end_ns: u128,
    pub queue_submit_ns: u128,
    pub poll_ns: u128,
    pub fence_wait_ns: u128,
    pub retire_ns: u128,
    pub submit_calls: u64,
    pub poll_calls: u64,
    pub fence_wait_calls: u64,
    pub retire_calls: u64,
    pub output_readback_enabled: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessedFrameOutput {
    pub token: FrameToken,
    pub presentation_position: u64,
    pub slot_index: usize,
    pub submission_sequence: u64,
    pub width: u32,
    pub height: u32,
    pub rgba_u16: Vec<u16>,
    pub checksum: u64,
    pub cpu_surface_bytes: usize,
    pub staging_bytes: usize,
    pub gpu_plane_bytes: usize,
    pub output_bytes: usize,
}

#[derive(Debug)]
pub enum FrameProcessorError {
    Resource(ResourceError),
    NoFrameSlotAvailable,
    UnknownFrameToken,
    FrameNotComplete,
    InvalidFrameState,
}

impl From<ResourceError> for FrameProcessorError {
    fn from(value: ResourceError) -> Self {
        Self::Resource(value)
    }
}

impl std::fmt::Display for FrameProcessorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Resource(err) => write!(f, "resource error: {err:?}"),
            Self::NoFrameSlotAvailable => write!(f, "no GPU frame slot available"),
            Self::UnknownFrameToken => write!(f, "unknown frame token"),
            Self::FrameNotComplete => write!(f, "frame is not complete"),
            Self::InvalidFrameState => write!(f, "invalid frame state transition"),
        }
    }
}

impl std::error::Error for FrameProcessorError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Yuv422P10Layout {
    y_samples: usize,
    chroma_samples: usize,
    output_samples: usize,
    cpu_surface_bytes: usize,
    staging_bytes: usize,
    gpu_plane_bytes: usize,
    output_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
struct Nv12Layout {
    y_bytes: usize,
    uv_bytes: usize,
    y_stride_bytes: usize,
    uv_stride_bytes: usize,
    output_samples: usize,
    cpu_surface_bytes: usize,
    staging_bytes: usize,
    gpu_plane_bytes: usize,
    output_bytes: usize,
}

#[derive(Debug)]
pub struct VulkanDeviceDiscovery {
    _instance: Arc<GpuInstance>,
    devices: Vec<RegisteredDevice>,
}

impl VulkanDeviceDiscovery {
    pub fn new() -> Result<Self, DeviceDiscoveryError> {
        let instance = Arc::new(GpuInstance::new().map_err(|err| {
            eprintln!("vulkan instance creation failed: {err:?}");
            DeviceDiscoveryError::BackendUnavailable
        })?);
        let physical_devices = instance.enumerate_physical_devices().map_err(|err| {
            eprintln!("vulkan physical-device enumeration failed: {err:?}");
            DeviceDiscoveryError::BackendFailed
        })?;

        let mut devices = Vec::new();
        for (index, physical_device) in physical_devices.into_iter().enumerate() {
            if devices.len() >= MAX_DEVICE_COUNT {
                break;
            }

            let properties = instance.physical_device_properties(physical_device);
            let memory_properties = instance.physical_device_memory_properties(physical_device);
            let queue_family_properties =
                instance.physical_device_queue_family_properties(physical_device);
            let queue_family_index = select_queue_family(&queue_family_properties)
                .ok_or(DeviceDiscoveryError::BackendFailed)?;
            let desc = describe_device((index as u64) + 1, properties)?;
            let extension_support = DeviceExtensionSupport::query(&instance, physical_device)
                .map_err(|err| {
                    eprintln!("vulkan extension query failed: {err:?}");
                    DeviceDiscoveryError::BackendFailed
                })?;
            let device = Arc::new(
                GpuDevice::new(
                    instance.clone(),
                    physical_device,
                    properties,
                    memory_properties,
                    queue_family_properties,
                    queue_family_index,
                    extension_support,
                )
                .map_err(|err| {
                    eprintln!("vulkan logical-device creation failed: {err:?}");
                    DeviceDiscoveryError::BackendFailed
                })?,
            );

            devices.push(RegisteredDevice { desc, device });
        }

        Ok(Self {
            _instance: instance,
            devices,
        })
    }

    pub fn import_and_validate_external_buffer(
        &self,
        source_device: &DeviceDesc,
        metadata: &ExportedResourceMetadata,
        handle: File,
    ) -> Result<(), ResourceError> {
        if metadata.device_id != source_device.id
            || metadata.kind != ResourceKind::Buffer
            || metadata.size_bytes == 0
            || metadata.allocation_size_bytes < metadata.size_bytes
            || metadata.attachment_count != 1
            || !metadata.selected_memory.host_visible
            || !metadata.selected_memory.host_coherent
        {
            return Err(ResourceError::ExportFailed);
        }

        let device = self.match_export_device(source_device)?;
        let usage = map_buffer_usage(metadata.buffer_usage.ok_or(ResourceError::ExportFailed)?)?;
        let handle_type = map_external_memory_handle_type(metadata.handle_type)?;
        validate_external_buffer_support(&device, usage, handle_type)?;
        let imported = import_external_buffer(&device, metadata, usage, handle_type, handle)?;
        let read = imported.read_bytes(0, SHARED_VALIDATION_MARKER.len())?;

        if read.as_slice() == SHARED_VALIDATION_MARKER {
            Ok(())
        } else {
            Err(ResourceError::ExportFailed)
        }
    }

    pub fn process_yuv422p10_surface(
        &self,
        upload: &Yuv422P10Upload<'_>,
    ) -> Result<Yuv422P10GpuOutput, ResourceError> {
        let mut processor = GpuFrameProcessor::new(
            self,
            GpuFrameProcessorConfig {
                device_id: upload.device_id,
                width: upload.width,
                height: upload.height,
                slot_count: 1,
                conversion: upload.conversion,
            },
        )
        .map_err(frame_processor_resource_error)?;
        let token = processor
            .submit_frame(
                upload,
                FrameIdentity {
                    presentation_position: 0,
                },
            )
            .map_err(frame_processor_resource_error)?;
        let output = processor
            .wait_for_frame(token)
            .map_err(frame_processor_resource_error)?;
        Ok(Yuv422P10GpuOutput {
            width: output.width,
            height: output.height,
            rgba_u16: output.rgba_u16,
            checksum: output.checksum,
            cpu_surface_bytes: output.cpu_surface_bytes,
            staging_bytes: output.staging_bytes,
            gpu_plane_bytes: output.gpu_plane_bytes,
            output_bytes: output.output_bytes,
        })
    }

    pub fn process_nv12_surface(
        &self,
        upload: &Nv12Upload<'_>,
    ) -> Result<Nv12GpuOutput, ResourceError> {
        let mut processor = Nv12FrameProcessor::new(
            self,
            Nv12FrameProcessorConfig {
                device_id: upload.device_id,
                coded_width: upload.coded_width,
                coded_height: upload.coded_height,
                visible_width: upload.visible_width,
                visible_height: upload.visible_height,
                slot_count: 1,
                conversion: upload.conversion,
                validation_readback: true,
            },
        )
        .map_err(frame_processor_resource_error)?;
        let token = processor
            .submit_frame(
                upload,
                FrameIdentity {
                    presentation_position: 0,
                },
            )
            .map_err(frame_processor_resource_error)?;
        let output = processor
            .wait_for_frame(token)
            .map_err(frame_processor_resource_error)?;
        Ok(Nv12GpuOutput {
            width: output.width,
            height: output.height,
            rgba_u16: output.rgba_u16,
            checksum: output.checksum,
            cpu_surface_bytes: output.cpu_surface_bytes,
            staging_bytes: output.staging_bytes,
            gpu_plane_bytes: output.gpu_plane_bytes,
            output_bytes: output.output_bytes,
        })
    }

    pub fn import_wait_and_validate_synced_buffer(
        &self,
        source_device: &DeviceDesc,
        resource_metadata: &ExportedResourceMetadata,
        resource_handle: File,
        sync_metadata: &ExportedSyncMetadata,
        sync_handle: File,
    ) -> Result<(), ResourceError> {
        if sync_metadata.handle_type != SyncExportHandleType::SyncFd
            || sync_metadata.attachment_count != 1
            || resource_metadata.device_id != source_device.id
            || resource_metadata.kind != ResourceKind::Buffer
            || resource_metadata.size_bytes == 0
            || resource_metadata.allocation_size_bytes < resource_metadata.size_bytes
            || resource_metadata.attachment_count != 1
        {
            return Err(ResourceError::ExportFailed);
        }

        let device = self.match_export_device(source_device)?;
        let usage = map_buffer_usage(
            resource_metadata
                .buffer_usage
                .ok_or(ResourceError::ExportFailed)?,
        )?;
        let handle_type = map_external_memory_handle_type(resource_metadata.handle_type)?;
        validate_external_buffer_support(&device, usage, handle_type)?;
        validate_sync_fd_support(&device).map_err(|_| ResourceError::ExportFailed)?;

        let imported_buffer = import_external_buffer(
            &device,
            resource_metadata,
            usage,
            handle_type,
            resource_handle,
        )?;
        let imported_semaphore = GpuSemaphore::new(device.clone(), false).map_err(|err| {
            eprintln!("vulkan sync semaphore creation failed: {err:?}");
            ResourceError::ExportFailed
        })?;
        imported_semaphore
            .import_sync_fd(sync_handle)
            .map_err(|err| {
                eprintln!("vulkan sync-fd import failed: {err:?}");
                ResourceError::ExportFailed
            })?;

        validate_synced_gpu_copy(
            &device,
            &imported_buffer,
            &imported_semaphore,
            sync_metadata.fill_pattern,
        )
    }

    pub fn import_wait_and_run_compute_increment_proof(
        &self,
        source_device: &DeviceDesc,
        resource_metadata: &ExportedResourceMetadata,
        resource_handle: File,
        sync_metadata: &ExportedSyncMetadata,
        sync_handle: File,
        input_values: &[u32],
    ) -> Result<Vec<u32>, ResourceError> {
        if input_values.is_empty()
            || !input_values
                .len()
                .is_multiple_of(COMPUTE_LOCAL_SIZE_X as usize)
        {
            return Err(ResourceError::InvalidBufferSize);
        }
        let input_bytes = u64::try_from(input_values.len())
            .ok()
            .and_then(|len| len.checked_mul(std::mem::size_of::<u32>() as u64))
            .ok_or(ResourceError::InvalidBufferSize)?;
        if input_bytes > resource_metadata.size_bytes
            || sync_metadata.handle_type != SyncExportHandleType::SyncFd
            || sync_metadata.attachment_count != 1
            || resource_metadata.device_id != source_device.id
            || resource_metadata.kind != ResourceKind::Buffer
            || resource_metadata.attachment_count != 1
            || !resource_metadata
                .buffer_usage
                .ok_or(ResourceError::ExportFailed)?
                .contains(BufferUsageFlags::STORAGE)
        {
            return Err(ResourceError::ExportFailed);
        }

        let device = self.match_export_device(source_device)?;
        if !device.queue_supports_compute() {
            return Err(ResourceError::UnsupportedMemoryRequirements);
        }
        let usage = map_buffer_usage(
            resource_metadata
                .buffer_usage
                .ok_or(ResourceError::ExportFailed)?,
        )?;
        let handle_type = map_external_memory_handle_type(resource_metadata.handle_type)?;
        validate_external_buffer_support(&device, usage, handle_type)?;
        validate_sync_fd_support(&device).map_err(|_| ResourceError::ExportFailed)?;

        let imported_buffer = import_external_buffer(
            &device,
            resource_metadata,
            usage,
            handle_type,
            resource_handle,
        )?;
        let imported_semaphore = GpuSemaphore::new(device.clone(), false).map_err(|err| {
            eprintln!("vulkan sync semaphore creation failed: {err:?}");
            ResourceError::ExportFailed
        })?;
        imported_semaphore
            .import_sync_fd(sync_handle)
            .map_err(|err| {
                eprintln!("vulkan sync-fd import failed: {err:?}");
                ResourceError::ExportFailed
            })?;

        run_compute_increment_proof(
            &device,
            &imported_buffer,
            &imported_semaphore,
            input_values,
            input_bytes,
        )
    }

    pub fn import_wait_and_run_image_invert_proof(
        &self,
        source_device: &DeviceDesc,
        resource_metadata: &ExportedResourceMetadata,
        resource_handle: File,
        sync_metadata: &ExportedSyncMetadata,
        sync_handle: File,
        input_pixels: &[u8],
    ) -> Result<Vec<u8>, ResourceError> {
        if sync_metadata.handle_type != SyncExportHandleType::SyncFd
            || sync_metadata.attachment_count != 1
            || resource_metadata.kind != ResourceKind::Image
            || resource_metadata.device_id != source_device.id
            || resource_metadata.attachment_count != 1
        {
            return Err(ResourceError::ExportFailed);
        }
        let width = resource_metadata
            .image_width
            .ok_or(ResourceError::ExportFailed)?;
        let height = resource_metadata
            .image_height
            .ok_or(ResourceError::ExportFailed)?;
        if !width.is_multiple_of(8) || !height.is_multiple_of(8) {
            return Err(ResourceError::InvalidImageDimensions);
        }
        let format = resource_metadata
            .pixel_format
            .ok_or(ResourceError::ExportFailed)?;
        let image_usage = resource_metadata
            .image_usage
            .ok_or(ResourceError::ExportFailed)?;
        let expected_len =
            qgs_protocol::image_byte_len(width, height, format).map_err(ResourceError::from)?;
        if input_pixels.len() as u64 != expected_len {
            return Err(ResourceError::InvalidImageDimensions);
        }

        let device = self.match_export_device(source_device)?;
        if !device.queue_supports_compute() {
            return Err(ResourceError::UnsupportedMemoryRequirements);
        }
        let vk_format = map_pixel_format(format)?;
        let usage = map_image_usage(image_usage)?;
        let handle_type = map_external_memory_handle_type(resource_metadata.handle_type)?;
        let tiling = if resource_metadata.handle_type == ExternalHandleType::DmaBuf {
            vk::ImageTiling::LINEAR
        } else {
            vk::ImageTiling::OPTIMAL
        };
        validate_external_image_support(&device, vk_format, usage, tiling, handle_type)?;
        validate_sync_fd_support(&device).map_err(|_| ResourceError::ExportFailed)?;

        let imported_image = import_external_image(
            &device,
            ImportedImageInfo {
                metadata: resource_metadata,
                format: vk_format,
                usage,
                tiling,
                handle_type,
                handle: resource_handle,
            },
        )?;
        let imported_semaphore = GpuSemaphore::new(device.clone(), false).map_err(|err| {
            eprintln!("vulkan sync semaphore creation failed: {err:?}");
            ResourceError::ExportFailed
        })?;
        imported_semaphore
            .import_sync_fd(sync_handle)
            .map_err(|err| {
                eprintln!("vulkan sync-fd import failed: {err:?}");
                ResourceError::ExportFailed
            })?;

        run_image_invert_proof(&device, &imported_image, &imported_semaphore, input_pixels)
    }

    fn match_export_device(
        &self,
        source_device: &DeviceDesc,
    ) -> Result<Arc<GpuDevice>, ResourceError> {
        self.devices
            .iter()
            .find(|device| device_matches_export_source(&device.desc, source_device))
            .map(|device| device.device.clone())
            .ok_or(ResourceError::UnknownDeviceId)
    }
}

impl DeviceDiscovery for VulkanDeviceDiscovery {
    fn enumerate_devices(&self) -> Result<Vec<DeviceDesc>, DeviceDiscoveryError> {
        Ok(self
            .devices
            .iter()
            .map(|device| device.desc.clone())
            .collect())
    }

    fn query_device_capabilities(
        &self,
        device_id: DeviceId,
    ) -> Result<DeviceCapabilities, DeviceDiscoveryError> {
        let device = self
            .devices
            .iter()
            .find(|device| device.desc.id == device_id)
            .ok_or(DeviceDiscoveryError::UnknownDeviceId)?;

        Ok(describe_capabilities(device.desc.id, &device.device))
    }
}

impl VideoCapabilityDiscovery for VulkanDeviceDiscovery {
    fn query_video_capabilities(
        &self,
        device_id: DeviceId,
    ) -> Result<VideoCapabilities, VideoCapabilityDiscoveryError> {
        self.devices
            .iter()
            .find(|device| device.desc.id == device_id)
            .ok_or(VideoCapabilityDiscoveryError::UnknownDeviceId)?;

        Ok(VideoCapabilities {
            device_id,
            decode: Vec::new(),
        })
    }
}

impl ResourceBackend for VulkanDeviceDiscovery {
    fn create_buffer(&self, desc: &BufferDesc) -> Result<BackendBufferAllocation, ResourceError> {
        desc.validate()?;
        let device = self
            .devices
            .iter()
            .find(|device| device.desc.id == desc.device_id)
            .ok_or(ResourceError::UnknownDeviceId)?
            .device
            .clone();
        let usage = map_buffer_usage(desc.usage)?;
        let external_handle_type = match desc.external_sharing {
            ExternalSharing::None => None,
            ExternalSharing::Required { handle_type } => {
                let handle_type = map_external_memory_handle_type(handle_type)?;
                validate_external_buffer_support(&device, usage, handle_type)?;
                Some(handle_type)
            }
        };
        let buffer = GpuBuffer::new(
            device.clone(),
            desc.size_bytes,
            usage,
            external_handle_type,
            required_memory_flags(desc),
            preferred_memory_flags(desc),
            false,
        )?;
        let selected_memory = selected_memory_properties(&device, buffer.memory_type_index)?;
        Ok(BackendBufferAllocation {
            resource: Box::new(VulkanBufferResource {
                buffer,
                desc: desc.clone(),
                selected_memory,
            }),
            selected_memory,
        })
    }

    fn create_image(&self, desc: &ImageDesc) -> Result<BackendImageAllocation, ResourceError> {
        desc.validate()?;
        let device = self
            .devices
            .iter()
            .find(|device| device.desc.id == desc.device_id)
            .ok_or(ResourceError::UnknownDeviceId)?
            .device
            .clone();
        let format = map_pixel_format(desc.format)?;
        let usage = map_image_usage(desc.usage)?;
        let (external_handle_type, tiling) = match desc.external_sharing {
            ExternalSharing::None => (None, vk::ImageTiling::OPTIMAL),
            ExternalSharing::Required { handle_type } => {
                let handle_type = map_external_memory_handle_type(handle_type)?;
                let tiling = if handle_type == vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT {
                    vk::ImageTiling::LINEAR
                } else {
                    vk::ImageTiling::OPTIMAL
                };
                validate_external_image_support(&device, format, usage, tiling, handle_type)?;
                (Some(handle_type), tiling)
            }
        };
        let image = GpuImage::new(
            device.clone(),
            ImageCreateParams {
                format,
                extent: [desc.width, desc.height, 1],
                usage,
                tiling,
                external_handle_type,
                required: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                preferred: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            },
        )?;
        let selected_memory = selected_memory_properties(&device, image.memory_type_index)?;
        Ok(BackendImageAllocation {
            resource: Box::new(VulkanImageResource {
                image,
                desc: desc.clone(),
                selected_memory,
            }),
            selected_memory,
        })
    }
}

impl SyncBackend for VulkanDeviceDiscovery {
    fn create_sync(&self, request: &CreateSyncRequest) -> Result<Box<dyn BackendSync>, SyncError> {
        if request.kind != SyncKind::BinarySemaphore {
            return Err(SyncError::SyncExportFailed);
        }
        if request.handle_type != SyncExportHandleType::SyncFd {
            return Err(SyncError::UnsupportedSyncHandleType);
        }

        let device = self
            .devices
            .iter()
            .find(|device| device.desc.id == request.device_id)
            .ok_or(SyncError::UnknownDeviceId)?
            .device
            .clone();
        validate_sync_fd_support(&device)?;
        let semaphore = GpuSemaphore::new(device, true).map_err(|err| {
            eprintln!("vulkan sync semaphore creation failed: {err:?}");
            SyncError::SyncExportFailed
        })?;

        Ok(Box::new(VulkanSyncResource {
            device_id: request.device_id,
            handle_type: request.handle_type,
            semaphore,
            exported: AtomicBool::new(false),
            pending: Mutex::new(Vec::new()),
        }))
    }
}

#[derive(Debug)]
struct RegisteredDevice {
    desc: DeviceDesc,
    device: Arc<GpuDevice>,
}

#[derive(Debug)]
struct VulkanBufferResource {
    buffer: GpuBuffer,
    desc: BufferDesc,
    selected_memory: SelectedMemoryProperties,
}

#[derive(Debug)]
struct VulkanImageResource {
    image: GpuImage,
    desc: ImageDesc,
    selected_memory: SelectedMemoryProperties,
}

#[derive(Debug)]
struct VulkanSyncResource {
    device_id: DeviceId,
    handle_type: SyncExportHandleType,
    semaphore: GpuSemaphore,
    exported: AtomicBool,
    pending: Mutex<Vec<PendingSubmission>>,
}

impl Drop for VulkanSyncResource {
    fn drop(&mut self) {
        if let Ok(mut pending) = self.pending.lock() {
            pending.clear();
        }
    }
}

impl BackendSync for VulkanSyncResource {
    fn export_for_resource(
        &self,
        request: &ExportSyncRequest,
        resource: &dyn BackendResource,
    ) -> Result<BackendSyncExport, SyncError> {
        if self.handle_type != SyncExportHandleType::SyncFd {
            return Err(SyncError::UnsupportedSyncHandleType);
        }
        if self.exported.swap(true, Ordering::AcqRel) {
            return Err(SyncError::SyncExportFailed);
        }

        let pending = if let Some(buffer) = resource.as_any().downcast_ref::<VulkanBufferResource>()
        {
            if buffer.desc.device_id != self.device_id {
                return Err(SyncError::UnknownResource);
            }
            submit_gpu_fill(&buffer.buffer, &self.semaphore, request.fill_pattern)?
        } else if let Some(image) = resource.as_any().downcast_ref::<VulkanImageResource>() {
            if image.desc.device_id != self.device_id {
                return Err(SyncError::UnknownResource);
            }
            submit_gpu_image_clear(&image.image, &self.semaphore, request.fill_pattern)?
        } else {
            return Err(SyncError::SyncExportFailed);
        };
        self.pending
            .lock()
            .map_err(|_| SyncError::SyncExportFailed)?
            .push(pending);

        let handle = self.semaphore.export_sync_fd().map_err(|err| {
            eprintln!("vulkan sync-fd export failed: {err:?}");
            SyncError::SyncExportFailed
        })?;

        Ok(BackendSyncExport {
            metadata: ExportedSyncMetadata {
                sync_id: request.sync_id,
                handle_type: self.handle_type,
                attachment_count: 1,
                fill_pattern: request.fill_pattern,
            },
            handle,
        })
    }
}

impl BackendResource for VulkanBufferResource {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn export(
        &self,
        request: &ExportResourceRequest,
    ) -> Result<BackendResourceExport, ResourceError> {
        let ExternalSharing::Required { handle_type } = self.desc.external_sharing else {
            return Err(ResourceError::ResourceNotExportable);
        };
        if request.handle_type != handle_type {
            return Err(ResourceError::UnsupportedExternalHandleType);
        }

        self.buffer.write_bytes(0, SHARED_VALIDATION_MARKER)?;
        let vk_handle_type = map_external_memory_handle_type(handle_type)?;
        let handle = self.buffer.memory.export_fd(vk_handle_type)?;

        Ok(BackendResourceExport {
            metadata: ExportedResourceMetadata {
                resource_id: request.resource_id,
                device_id: self.desc.device_id,
                kind: ResourceKind::Buffer,
                size_bytes: self.desc.size_bytes,
                allocation_size_bytes: self.buffer.memory.size,
                buffer_usage: Some(self.desc.usage),
                image_width: None,
                image_height: None,
                pixel_format: None,
                image_usage: None,
                backend_memory_type_index: self.buffer.memory_type_index,
                backend_image_layout_token: None,
                handle_type,
                selected_memory: self.selected_memory,
                dedicated_allocation: true,
                attachment_count: 1,
            },
            handle,
        })
    }
}

impl BackendResource for VulkanImageResource {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn export(
        &self,
        request: &ExportResourceRequest,
    ) -> Result<BackendResourceExport, ResourceError> {
        let ExternalSharing::Required { handle_type } = self.desc.external_sharing else {
            return Err(ResourceError::ResourceNotExportable);
        };
        if request.handle_type != handle_type {
            return Err(ResourceError::UnsupportedExternalHandleType);
        }

        let vk_handle_type = map_external_memory_handle_type(handle_type)?;
        let handle = self.image.memory.export_fd(vk_handle_type)?;

        Ok(BackendResourceExport {
            metadata: ExportedResourceMetadata {
                resource_id: request.resource_id,
                device_id: self.desc.device_id,
                kind: ResourceKind::Image,
                size_bytes: qgs_protocol::image_byte_len(
                    self.desc.width,
                    self.desc.height,
                    self.desc.format,
                )?,
                allocation_size_bytes: self.image.memory.size,
                buffer_usage: None,
                image_width: Some(self.desc.width),
                image_height: Some(self.desc.height),
                pixel_format: Some(self.desc.format),
                image_usage: Some(self.desc.usage),
                backend_memory_type_index: self.image.memory_type_index,
                backend_image_layout_token: None,
                handle_type,
                selected_memory: self.selected_memory,
                dedicated_allocation: true,
                attachment_count: 1,
            },
            handle,
        })
    }
}

struct GpuInstance {
    _entry: ash::Entry,
    instance: ash::Instance,
}

impl std::fmt::Debug for GpuInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuInstance").finish_non_exhaustive()
    }
}

impl GpuInstance {
    fn new() -> BackendResult<Self> {
        let entry = ash::Entry::linked();
        let app_name = c"qgs-vulkan";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(1)
            .engine_name(app_name)
            .engine_version(1)
            .api_version(vk::make_api_version(0, 1, 2, 0));
        let validation_layer = c"VK_LAYER_KHRONOS_validation";
        let layer_names =
            if validation_requested() && validation_layer_available(&entry, validation_layer)? {
                vec![validation_layer.as_ptr()]
            } else {
                Vec::new()
            };
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names);
        // SAFETY: The create info references static C string literals and no
        // enabled extension/layer arrays. The returned instance is owned by
        // GpuInstance and destroyed in Drop.
        let instance = unsafe { entry.create_instance(&create_info, None) }?;
        Ok(Self {
            _entry: entry,
            instance,
        })
    }

    fn enumerate_physical_devices(&self) -> BackendResult<Vec<vk::PhysicalDevice>> {
        // SAFETY: self.instance is a live Vulkan instance.
        unsafe { self.instance.enumerate_physical_devices() }.map_err(BackendError::from)
    }

    fn physical_device_properties(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> vk::PhysicalDeviceProperties {
        // SAFETY: physical_device was returned by this instance during enumeration.
        unsafe {
            self.instance
                .get_physical_device_properties(physical_device)
        }
    }

    fn physical_device_memory_properties(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> vk::PhysicalDeviceMemoryProperties {
        // SAFETY: physical_device was returned by this instance during enumeration.
        unsafe {
            self.instance
                .get_physical_device_memory_properties(physical_device)
        }
    }

    fn physical_device_queue_family_properties(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Vec<vk::QueueFamilyProperties> {
        // SAFETY: physical_device was returned by this instance during enumeration.
        unsafe {
            self.instance
                .get_physical_device_queue_family_properties(physical_device)
        }
    }
}

fn validation_requested() -> bool {
    matches!(
        std::env::var("QGS_VULKAN_ENABLE_VALIDATION").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn validation_layer_available(entry: &ash::Entry, name: &CStr) -> BackendResult<bool> {
    // SAFETY: entry is live and the loader writes the returned layer-property
    // vector. QGS only reads fixed null-terminated layer_name fields.
    let layers = unsafe { entry.enumerate_instance_layer_properties() }?;
    Ok(layers.iter().any(|layer| {
        // SAFETY: Vulkan layer_name fields are fixed-size null-terminated strings.
        let layer_name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
        layer_name == name
    }))
}

impl Drop for GpuInstance {
    fn drop(&mut self) {
        // SAFETY: GpuInstance owns this instance. GpuDevice holds an Arc back to
        // the instance, so devices are dropped before the final instance Arc.
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct DeviceExtensionSupport {
    khr_external_memory: bool,
    khr_external_memory_fd: bool,
    ext_external_memory_dma_buf: bool,
    khr_dedicated_allocation: bool,
    khr_external_semaphore: bool,
    khr_external_semaphore_fd: bool,
    khr_external_fence_fd: bool,
    ext_image_drm_format_modifier: bool,
}

impl DeviceExtensionSupport {
    fn query(instance: &GpuInstance, physical_device: vk::PhysicalDevice) -> BackendResult<Self> {
        // SAFETY: physical_device was returned by this live instance.
        let properties = unsafe {
            instance
                .instance
                .enumerate_device_extension_properties(physical_device)
        }?;
        let has = |name: &CStr| {
            properties.iter().any(|extension| {
                // SAFETY: Vulkan extension_name fields are fixed-size null-terminated strings.
                let extension_name = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };
                extension_name == name
            })
        };
        Ok(Self {
            khr_external_memory: has(c"VK_KHR_external_memory"),
            khr_external_memory_fd: has(ash::khr::external_memory_fd::NAME),
            ext_external_memory_dma_buf: has(c"VK_EXT_external_memory_dma_buf"),
            khr_dedicated_allocation: has(c"VK_KHR_dedicated_allocation"),
            khr_external_semaphore: has(c"VK_KHR_external_semaphore"),
            khr_external_semaphore_fd: has(ash::khr::external_semaphore_fd::NAME),
            khr_external_fence_fd: has(c"VK_KHR_external_fence_fd"),
            ext_image_drm_format_modifier: has(c"VK_EXT_image_drm_format_modifier"),
        })
    }
}

struct GpuDevice {
    instance: Arc<GpuInstance>,
    physical_device: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    queue_family_properties: Vec<vk::QueueFamilyProperties>,
    extension_support: DeviceExtensionSupport,
    device: ash::Device,
    external_memory_fd: ash::khr::external_memory_fd::Device,
    external_semaphore_fd: ash::khr::external_semaphore_fd::Device,
    queue: vk::Queue,
    queue_family_index: u32,
}

impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevice")
            .field("physical_device", &self.physical_device)
            .field("queue_family_index", &self.queue_family_index)
            .field("extension_support", &self.extension_support)
            .finish_non_exhaustive()
    }
}

impl GpuDevice {
    fn new(
        instance: Arc<GpuInstance>,
        physical_device: vk::PhysicalDevice,
        properties: vk::PhysicalDeviceProperties,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        queue_family_properties: Vec<vk::QueueFamilyProperties>,
        queue_family_index: u32,
        extension_support: DeviceExtensionSupport,
    ) -> BackendResult<Self> {
        let priorities = [1.0_f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];
        let mut extension_names = Vec::new();
        if extension_support.khr_external_memory {
            extension_names.push(c"VK_KHR_external_memory".as_ptr());
        }
        if extension_support.khr_external_memory_fd {
            extension_names.push(ash::khr::external_memory_fd::NAME.as_ptr());
        }
        if extension_support.ext_external_memory_dma_buf {
            extension_names.push(c"VK_EXT_external_memory_dma_buf".as_ptr());
        }
        if extension_support.khr_dedicated_allocation {
            extension_names.push(c"VK_KHR_dedicated_allocation".as_ptr());
        }
        if extension_support.khr_external_semaphore {
            extension_names.push(c"VK_KHR_external_semaphore".as_ptr());
        }
        if extension_support.khr_external_semaphore_fd {
            extension_names.push(ash::khr::external_semaphore_fd::NAME.as_ptr());
        }
        if extension_support.ext_image_drm_format_modifier {
            extension_names.push(c"VK_EXT_image_drm_format_modifier".as_ptr());
        }
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&extension_names);
        // SAFETY: physical_device and queue_family_index were queried from
        // instance. extension_names contains only device-supported extensions.
        let device = unsafe {
            instance
                .instance
                .create_device(physical_device, &create_info, None)
        }?;
        // SAFETY: queue index 0 exists because the selected family reported a
        // non-zero queue count and was used to create the logical device.
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let external_memory_fd =
            ash::khr::external_memory_fd::Device::new(&instance.instance, &device);
        let external_semaphore_fd =
            ash::khr::external_semaphore_fd::Device::new(&instance.instance, &device);

        Ok(Self {
            instance,
            physical_device,
            properties,
            memory_properties,
            queue_family_properties,
            extension_support,
            device,
            external_memory_fd,
            external_semaphore_fd,
            queue,
            queue_family_index,
        })
    }

    fn queue_supports_compute(&self) -> bool {
        self.queue_family_properties
            .get(self.queue_family_index as usize)
            .is_some_and(|queue| queue.queue_flags.contains(vk::QueueFlags::COMPUTE))
    }

    fn choose_memory_type(
        &self,
        memory_type_bits: u32,
        required: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        choose_memory_type_from_properties(
            &self.memory_properties,
            memory_type_bits,
            required,
            preferred,
        )
    }

    fn memory_fd_type_bits(
        &self,
        handle_type: vk::ExternalMemoryHandleTypeFlags,
        file: &File,
    ) -> BackendResult<u32> {
        let mut properties = vk::MemoryFdPropertiesKHR::default();
        // SAFETY: file.as_raw_fd() borrows the fd for the duration of the call.
        // vkGetMemoryFdPropertiesKHR does not consume POSIX fd ownership; the
        // caller retains the File for the later import operation.
        unsafe {
            self.external_memory_fd.get_memory_fd_properties(
                handle_type,
                file.as_raw_fd(),
                &mut properties,
            )
        }?;
        Ok(properties.memory_type_bits)
    }
}

impl Drop for GpuDevice {
    fn drop(&mut self) {
        // SAFETY: GpuDevice owns this logical device. Resource wrappers hold an
        // Arc<GpuDevice>, so they are dropped before the final device owner.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
        }
    }
}

#[derive(Debug)]
struct GpuMemory {
    device: Arc<GpuDevice>,
    memory: vk::DeviceMemory,
    size: u64,
    memory_type_index: u32,
    mapped: Mutex<Option<usize>>,
}

impl GpuMemory {
    fn allocate(
        device: Arc<GpuDevice>,
        requirements: vk::MemoryRequirements,
        required: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
        export_handle_type: Option<vk::ExternalMemoryHandleTypeFlags>,
        dedicated: Option<DedicatedResource>,
    ) -> BackendResult<Self> {
        let memory_type_index = device
            .choose_memory_type(requirements.memory_type_bits, required, preferred)
            .ok_or(BackendError::NoMemoryType)?;
        let mut export_info = export_handle_type
            .map(|handle_types| vk::ExportMemoryAllocateInfo::default().handle_types(handle_types));
        let mut dedicated_info = dedicated.map(|dedicated| match dedicated {
            DedicatedResource::Buffer(buffer) => {
                vk::MemoryDedicatedAllocateInfo::default().buffer(buffer)
            }
            DedicatedResource::Image(image) => {
                vk::MemoryDedicatedAllocateInfo::default().image(image)
            }
        });
        let mut allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        if let Some(info) = export_info.as_mut() {
            allocate_info = allocate_info.push_next(info);
        }
        if let Some(info) = dedicated_info.as_mut() {
            allocate_info = allocate_info.push_next(info);
        }
        // SAFETY: allocation size and memory type index come from Vulkan memory
        // requirements and QGS memory selection. pNext structs live through call.
        let memory = unsafe { device.device.allocate_memory(&allocate_info, None) }?;
        Ok(Self {
            device,
            memory,
            size: requirements.size,
            memory_type_index,
            mapped: Mutex::new(None),
        })
    }

    fn import_fd(
        device: Arc<GpuDevice>,
        size: u64,
        memory_type_index: u32,
        handle_type: vk::ExternalMemoryHandleTypeFlags,
        file: File,
        dedicated: Option<DedicatedResource>,
    ) -> BackendResult<Self> {
        let raw_fd = file.into_raw_fd();
        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(handle_type)
            .fd(raw_fd);
        let mut dedicated_info = dedicated.map(|dedicated| match dedicated {
            DedicatedResource::Buffer(buffer) => {
                vk::MemoryDedicatedAllocateInfo::default().buffer(buffer)
            }
            DedicatedResource::Image(image) => {
                vk::MemoryDedicatedAllocateInfo::default().image(image)
            }
        });
        let mut allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type_index)
            .push_next(&mut import_info);
        if let Some(info) = dedicated_info.as_mut() {
            allocate_info = allocate_info.push_next(info);
        }
        // SAFETY: raw_fd is moved out of File and consumed by Vulkan on
        // successful import. On allocation failure Vulkan leaves ownership with
        // the application; QGS reconstructs File below so it can be closed.
        let memory = match unsafe { device.device.allocate_memory(&allocate_info, None) } {
            Ok(memory) => memory,
            Err(err) => {
                // SAFETY: Vulkan did not consume fd on allocation failure.
                unsafe {
                    drop(File::from_raw_fd(raw_fd));
                }
                return Err(BackendError::Vk(err));
            }
        };
        Ok(Self {
            device,
            memory,
            size,
            memory_type_index,
            mapped: Mutex::new(None),
        })
    }

    fn export_fd(
        &self,
        handle_type: vk::ExternalMemoryHandleTypeFlags,
    ) -> Result<File, ResourceError> {
        let info = vk::MemoryGetFdInfoKHR::default()
            .memory(self.memory)
            .handle_type(handle_type);
        // SAFETY: self.memory is a live memory object allocated with export
        // support for this handle type.
        let fd = unsafe { self.device.external_memory_fd.get_memory_fd(&info) }.map_err(|err| {
            eprintln!("vulkan memory fd export failed: {err:?}");
            ResourceError::ExportFailed
        })?;
        // SAFETY: vkGetMemoryFdKHR returns a new owned POSIX fd on success.
        Ok(unsafe { File::from_raw_fd(fd) })
    }

    fn map(&self) -> BackendResult<*mut std::ffi::c_void> {
        let mut mapped = self.mapped.lock().map_err(|_| BackendError::InvalidInput)?;
        if let Some(ptr) = *mapped {
            return Ok(ptr as *mut std::ffi::c_void);
        }
        // SAFETY: self.memory is live, offset 0 and size self.size are inside
        // the allocation, and QGS serializes mapping through mapped mutex.
        let ptr = unsafe {
            self.device
                .device
                .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
        }?;
        *mapped = Some(ptr as usize);
        Ok(ptr)
    }

    fn flush(&self, offset: u64, size: u64) -> BackendResult<()> {
        if self.is_host_coherent() {
            return Ok(());
        }
        let range = vk::MappedMemoryRange::default()
            .memory(self.memory)
            .offset(offset)
            .size(size);
        // SAFETY: range describes a currently mapped byte range in this memory.
        unsafe { self.device.device.flush_mapped_memory_ranges(&[range]) }?;
        Ok(())
    }

    fn invalidate(&self, offset: u64, size: u64) -> BackendResult<()> {
        if self.is_host_coherent() {
            return Ok(());
        }
        let range = vk::MappedMemoryRange::default()
            .memory(self.memory)
            .offset(offset)
            .size(size);
        // SAFETY: range describes a currently mapped byte range in this memory.
        unsafe { self.device.device.invalidate_mapped_memory_ranges(&[range]) }?;
        Ok(())
    }

    fn is_host_coherent(&self) -> bool {
        self.device.memory_properties.memory_types[self.memory_type_index as usize]
            .property_flags
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
    }
}

impl Drop for GpuMemory {
    fn drop(&mut self) {
        // SAFETY: GpuMemory owns this memory. Any mapping was created by this
        // wrapper and must be unmapped before freeing.
        unsafe {
            if self
                .mapped
                .lock()
                .ok()
                .and_then(|mut mapped| mapped.take())
                .is_some()
            {
                self.device.device.unmap_memory(self.memory);
            }
            self.device.device.free_memory(self.memory, None);
        }
    }
}

#[derive(Clone, Copy)]
enum DedicatedResource {
    Buffer(vk::Buffer),
    Image(vk::Image),
}

#[derive(Debug)]
struct GpuBuffer {
    device: Arc<GpuDevice>,
    buffer: vk::Buffer,
    memory: Arc<GpuMemory>,
    size: u64,
    memory_type_index: u32,
}

struct ImportedBufferInfo {
    size: u64,
    usage: vk::BufferUsageFlags,
    handle_type: vk::ExternalMemoryHandleTypeFlags,
    file: File,
    allocation_size: u64,
    memory_type_index: u32,
    dedicated: bool,
}

impl GpuBuffer {
    fn new(
        device: Arc<GpuDevice>,
        size: u64,
        usage: vk::BufferUsageFlags,
        external_handle_type: Option<vk::ExternalMemoryHandleTypeFlags>,
        required: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
        import_only: bool,
    ) -> Result<Self, ResourceError> {
        let mut external_info = external_handle_type.map(|handle_types| {
            vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_types)
        });
        let mut create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        if let Some(info) = external_info.as_mut() {
            create_info = create_info.push_next(info);
        }
        // SAFETY: create_info is fully initialized and pNext storage lives for
        // the duration of the call.
        let buffer = unsafe { device.device.create_buffer(&create_info, None) }.map_err(|err| {
            eprintln!("vulkan buffer creation failed: {err:?}");
            ResourceError::AllocationFailed
        })?;
        // SAFETY: buffer is live and was created by this device.
        let requirements = unsafe { device.device.get_buffer_memory_requirements(buffer) };
        let memory = GpuMemory::allocate(
            device.clone(),
            requirements,
            required,
            preferred,
            external_handle_type,
            Some(DedicatedResource::Buffer(buffer)),
        )
        .map_err(|err| {
            eprintln!("vulkan buffer memory allocation failed: {err:?}");
            // SAFETY: buffer was created by this device and has not been bound.
            unsafe {
                device.device.destroy_buffer(buffer, None);
            }
            ResourceError::AllocationFailed
        })?;
        // SAFETY: buffer and memory are from the same device. Offset 0 satisfies
        // Vulkan's reported alignment because it is always aligned.
        unsafe { device.device.bind_buffer_memory(buffer, memory.memory, 0) }.map_err(|err| {
            eprintln!("vulkan buffer memory bind failed: {err:?}");
            ResourceError::AllocationFailed
        })?;
        let memory_type_index = memory.memory_type_index;
        let memory = Arc::new(memory);
        let buffer = Self {
            device,
            buffer,
            memory,
            size,
            memory_type_index,
        };
        if !import_only && required.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            let _ = buffer.memory.map().map_err(|err| {
                eprintln!("vulkan buffer memory map failed: {err:?}");
                ResourceError::AllocationFailed
            })?;
        }
        Ok(buffer)
    }

    fn imported(device: Arc<GpuDevice>, info: ImportedBufferInfo) -> Result<Self, ResourceError> {
        let mut external_info =
            vk::ExternalMemoryBufferCreateInfo::default().handle_types(info.handle_type);
        let create_info = vk::BufferCreateInfo::default()
            .push_next(&mut external_info)
            .size(info.size)
            .usage(info.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: create_info is fully initialized and pNext storage lives for
        // the duration of the call.
        let buffer = unsafe { device.device.create_buffer(&create_info, None) }.map_err(|err| {
            eprintln!("vulkan imported buffer creation failed: {err:?}");
            ResourceError::ExportFailed
        })?;
        let dedicated_resource = info.dedicated.then_some(DedicatedResource::Buffer(buffer));
        let memory = GpuMemory::import_fd(
            device.clone(),
            info.allocation_size,
            info.memory_type_index,
            info.handle_type,
            info.file,
            dedicated_resource,
        )
        .map_err(|err| {
            eprintln!("vulkan imported buffer memory import failed: {err:?}");
            // SAFETY: buffer was created by this device and has not been bound.
            unsafe {
                device.device.destroy_buffer(buffer, None);
            }
            ResourceError::ExportFailed
        })?;
        // SAFETY: buffer and memory are from the same device. Offset 0 satisfies
        // Vulkan's reported alignment because it is always aligned.
        unsafe { device.device.bind_buffer_memory(buffer, memory.memory, 0) }.map_err(|err| {
            eprintln!("vulkan imported buffer memory bind failed: {err:?}");
            ResourceError::ExportFailed
        })?;
        Ok(Self {
            device,
            buffer,
            memory: Arc::new(memory),
            size: info.size,
            memory_type_index: info.memory_type_index,
        })
    }

    fn write_bytes(&self, offset: u64, bytes: &[u8]) -> Result<(), ResourceError> {
        let size = u64::try_from(bytes.len()).map_err(|_| ResourceError::InvalidBufferSize)?;
        if offset.checked_add(size).is_none_or(|end| end > self.size) {
            return Err(ResourceError::InvalidBufferSize);
        }
        let ptr = self.memory.map().map_err(|err| {
            eprintln!("vulkan buffer map for write failed: {err:?}");
            ResourceError::UnsupportedMemoryRequirements
        })?;
        // SAFETY: offset..offset+size was bounds-checked above, and ptr points
        // to the mapped allocation for this buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                ptr.cast::<u8>().add(offset as usize),
                bytes.len(),
            );
        }
        self.memory.flush(offset, size).map_err(|err| {
            eprintln!("vulkan buffer flush failed: {err:?}");
            ResourceError::ExportFailed
        })
    }

    fn write_pitched_rows(
        &self,
        offset: u64,
        source: &[u8],
        source_stride: usize,
        row_bytes: usize,
        row_count: usize,
        destination_stride: usize,
    ) -> Result<(), ResourceError> {
        if row_bytes == 0
            || row_count == 0
            || source_stride < row_bytes
            || destination_stride < row_bytes
        {
            return Err(ResourceError::InvalidBufferSize);
        }
        let source_required = source_stride
            .checked_mul(row_count)
            .ok_or(ResourceError::InvalidBufferSize)?;
        if source.len() < source_required {
            return Err(ResourceError::InvalidBufferSize);
        }
        let destination_bytes = destination_stride
            .checked_mul(row_count)
            .ok_or(ResourceError::InvalidBufferSize)?;
        let destination_size =
            u64::try_from(destination_bytes).map_err(|_| ResourceError::InvalidBufferSize)?;
        if offset
            .checked_add(destination_size)
            .is_none_or(|end| end > self.size)
        {
            return Err(ResourceError::InvalidBufferSize);
        }
        let ptr = self.memory.map().map_err(|err| {
            eprintln!("vulkan buffer map for pitched write failed: {err:?}");
            ResourceError::UnsupportedMemoryRequirements
        })?;
        // SAFETY: destination range was bounds-checked against this buffer;
        // source rows were bounds-checked above; rows do not overlap in the
        // destination because destination_stride >= row_bytes.
        unsafe {
            let base = ptr.cast::<u8>().add(offset as usize);
            for row in 0..row_count {
                let src_offset = row * source_stride;
                let dst_offset = row * destination_stride;
                std::ptr::copy_nonoverlapping(
                    source.as_ptr().add(src_offset),
                    base.add(dst_offset),
                    row_bytes,
                );
                if destination_stride > row_bytes {
                    std::ptr::write_bytes(
                        base.add(dst_offset + row_bytes),
                        0,
                        destination_stride - row_bytes,
                    );
                }
            }
        }
        self.memory.flush(offset, destination_size).map_err(|err| {
            eprintln!("vulkan buffer pitched flush failed: {err:?}");
            ResourceError::ExportFailed
        })
    }

    fn read_bytes(&self, offset: u64, len: usize) -> Result<Vec<u8>, ResourceError> {
        let size = u64::try_from(len).map_err(|_| ResourceError::InvalidBufferSize)?;
        if offset.checked_add(size).is_none_or(|end| end > self.size) {
            return Err(ResourceError::InvalidBufferSize);
        }
        let ptr = self.memory.map().map_err(|err| {
            eprintln!("vulkan buffer map for read failed: {err:?}");
            ResourceError::UnsupportedMemoryRequirements
        })?;
        self.memory.invalidate(offset, size).map_err(|err| {
            eprintln!("vulkan buffer invalidate failed: {err:?}");
            ResourceError::ExportFailed
        })?;
        let mut out = vec![0_u8; len];
        // SAFETY: offset..offset+size was bounds-checked above, ptr points to
        // mapped allocation memory, and out is valid for len bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.cast::<u8>().add(offset as usize),
                out.as_mut_ptr(),
                len,
            );
        }
        Ok(out)
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        // SAFETY: GpuBuffer owns this VkBuffer. Bound memory is held by Arc and
        // remains alive until after the buffer is destroyed.
        unsafe {
            let _ = self.device.device.device_wait_idle();
            self.device.device.destroy_buffer(self.buffer, None);
        }
    }
}

#[derive(Debug)]
struct GpuImage {
    device: Arc<GpuDevice>,
    image: vk::Image,
    memory: Arc<GpuMemory>,
    format: vk::Format,
    extent: [u32; 3],
    memory_type_index: u32,
}

struct ImageCreateParams {
    format: vk::Format,
    extent: [u32; 3],
    usage: vk::ImageUsageFlags,
    tiling: vk::ImageTiling,
    external_handle_type: Option<vk::ExternalMemoryHandleTypeFlags>,
    required: vk::MemoryPropertyFlags,
    preferred: vk::MemoryPropertyFlags,
}

impl GpuImage {
    fn new(device: Arc<GpuDevice>, params: ImageCreateParams) -> Result<Self, ResourceError> {
        let mut external_info = params.external_handle_type.map(|handle_types| {
            vk::ExternalMemoryImageCreateInfo::default().handle_types(handle_types)
        });
        let mut create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(params.format)
            .extent(vk::Extent3D {
                width: params.extent[0],
                height: params.extent[1],
                depth: params.extent[2],
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(params.tiling)
            .usage(params.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        if let Some(info) = external_info.as_mut() {
            create_info = create_info.push_next(info);
        }
        // SAFETY: create_info is fully initialized and pNext storage lives for
        // the duration of the call.
        let image = unsafe { device.device.create_image(&create_info, None) }.map_err(|err| {
            eprintln!("vulkan image creation failed: {err:?}");
            ResourceError::AllocationFailed
        })?;
        // SAFETY: image is live and was created by this device.
        let requirements = unsafe { device.device.get_image_memory_requirements(image) };
        let memory = GpuMemory::allocate(
            device.clone(),
            requirements,
            params.required,
            params.preferred,
            params.external_handle_type,
            Some(DedicatedResource::Image(image)),
        )
        .map_err(|err| {
            eprintln!("vulkan image memory allocation failed: {err:?}");
            // SAFETY: image was created by this device and has not been bound.
            unsafe {
                device.device.destroy_image(image, None);
            }
            ResourceError::AllocationFailed
        })?;
        // SAFETY: image and memory are from the same device. Offset 0 satisfies
        // Vulkan's reported alignment because it is always aligned.
        unsafe { device.device.bind_image_memory(image, memory.memory, 0) }.map_err(|err| {
            eprintln!("vulkan image memory bind failed: {err:?}");
            ResourceError::AllocationFailed
        })?;
        let memory_type_index = memory.memory_type_index;
        Ok(Self {
            device,
            image,
            memory: Arc::new(memory),
            format: params.format,
            extent: params.extent,
            memory_type_index,
        })
    }

    fn imported(
        device: Arc<GpuDevice>,
        metadata: &ExportedResourceMetadata,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        tiling: vk::ImageTiling,
        handle_type: vk::ExternalMemoryHandleTypeFlags,
        file: File,
    ) -> Result<Self, ResourceError> {
        let width = metadata.image_width.ok_or(ResourceError::ExportFailed)?;
        let height = metadata.image_height.ok_or(ResourceError::ExportFailed)?;
        let mut external_info =
            vk::ExternalMemoryImageCreateInfo::default().handle_types(handle_type);
        let create_info = vk::ImageCreateInfo::default()
            .push_next(&mut external_info)
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(tiling)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: create_info is fully initialized and pNext storage lives for
        // the duration of the call.
        let image = unsafe { device.device.create_image(&create_info, None) }.map_err(|err| {
            eprintln!("vulkan imported image creation failed: {err:?}");
            ResourceError::ExportFailed
        })?;
        let dedicated = metadata
            .dedicated_allocation
            .then_some(DedicatedResource::Image(image));
        let memory = GpuMemory::import_fd(
            device.clone(),
            metadata.allocation_size_bytes,
            metadata.backend_memory_type_index,
            handle_type,
            file,
            dedicated,
        )
        .map_err(|err| {
            eprintln!("vulkan imported image memory import failed: {err:?}");
            // SAFETY: image was created by this device and has not been bound.
            unsafe {
                device.device.destroy_image(image, None);
            }
            ResourceError::ExportFailed
        })?;
        // SAFETY: image and memory are from the same device. Offset 0 satisfies
        // Vulkan's reported alignment because it is always aligned.
        unsafe { device.device.bind_image_memory(image, memory.memory, 0) }.map_err(|err| {
            eprintln!("vulkan imported image memory bind failed: {err:?}");
            ResourceError::ExportFailed
        })?;
        Ok(Self {
            device,
            image,
            memory: Arc::new(memory),
            format,
            extent: [width, height, 1],
            memory_type_index: metadata.backend_memory_type_index,
        })
    }

    fn create_view(&self) -> BackendResult<GpuImageView> {
        let create_info = vk::ImageViewCreateInfo::default()
            .image(self.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(self.format)
            .subresource_range(color_subresource_range());
        // SAFETY: image is live, format/view type match image creation, and
        // the view is owned by GpuImageView.
        let view = unsafe { self.device.device.create_image_view(&create_info, None) }?;
        Ok(GpuImageView {
            device: self.device.clone(),
            view,
        })
    }
}

impl Drop for GpuImage {
    fn drop(&mut self) {
        // SAFETY: GpuImage owns this VkImage. Bound memory is held by Arc and
        // remains alive until after the image is destroyed.
        unsafe {
            let _ = self.device.device.device_wait_idle();
            self.device.device.destroy_image(self.image, None);
        }
    }
}

#[derive(Debug)]
struct GpuImageView {
    device: Arc<GpuDevice>,
    view: vk::ImageView,
}

impl Drop for GpuImageView {
    fn drop(&mut self) {
        // SAFETY: GpuImageView owns this image view.
        unsafe {
            self.device.device.destroy_image_view(self.view, None);
        }
    }
}

#[derive(Debug)]
struct GpuSemaphore {
    device: Arc<GpuDevice>,
    semaphore: vk::Semaphore,
}

impl GpuSemaphore {
    fn new(device: Arc<GpuDevice>, export_sync_fd: bool) -> BackendResult<Self> {
        let mut export_info = export_sync_fd.then(|| {
            vk::ExportSemaphoreCreateInfo::default()
                .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
        });
        let mut create_info = vk::SemaphoreCreateInfo::default();
        if let Some(info) = export_info.as_mut() {
            create_info = create_info.push_next(info);
        }
        // SAFETY: create_info is valid and pNext storage lives for call.
        let semaphore = unsafe { device.device.create_semaphore(&create_info, None) }?;
        Ok(Self { device, semaphore })
    }

    fn export_sync_fd(&self) -> BackendResult<File> {
        let info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(self.semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        // SAFETY: semaphore was created with sync FD export support and has a
        // pending/available signal payload submitted by QGS before export.
        let fd = unsafe { self.device.external_semaphore_fd.get_semaphore_fd(&info) }?;
        // SAFETY: vkGetSemaphoreFdKHR returns a new owned POSIX fd on success.
        Ok(unsafe { File::from_raw_fd(fd) })
    }

    fn import_sync_fd(&self, file: File) -> BackendResult<()> {
        let raw_fd = file.into_raw_fd();
        let info = vk::ImportSemaphoreFdInfoKHR::default()
            .semaphore(self.semaphore)
            .flags(vk::SemaphoreImportFlags::TEMPORARY)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
            .fd(raw_fd);
        // SAFETY: raw_fd is moved from File and consumed by Vulkan on success.
        // On failure, Vulkan leaves ownership with QGS and the fd is closed.
        match unsafe { self.device.external_semaphore_fd.import_semaphore_fd(&info) } {
            Ok(()) => Ok(()),
            Err(err) => {
                // SAFETY: Vulkan did not consume fd on import failure.
                unsafe {
                    drop(File::from_raw_fd(raw_fd));
                }
                Err(BackendError::Vk(err))
            }
        }
    }
}

impl Drop for GpuSemaphore {
    fn drop(&mut self) {
        // SAFETY: GpuSemaphore owns this semaphore.
        unsafe {
            self.device.device.destroy_semaphore(self.semaphore, None);
        }
    }
}

#[derive(Debug)]
struct GpuFence {
    device: Arc<GpuDevice>,
    fence: vk::Fence,
}

impl GpuFence {
    fn new(device: Arc<GpuDevice>) -> BackendResult<Self> {
        // SAFETY: create info is valid and fence is owned by GpuFence.
        let fence = unsafe {
            device
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }?;
        Ok(Self { device, fence })
    }

    fn is_signaled(&self) -> BackendResult<bool> {
        // SAFETY: fence belongs to this device and is live.
        unsafe { self.device.device.get_fence_status(self.fence) }.map_err(BackendError::Vk)
    }

    fn reset(&self) -> BackendResult<()> {
        // SAFETY: callers reset only after the previous submission associated
        // with this fence has completed and the slot is available again.
        unsafe { self.device.device.reset_fences(&[self.fence]) }?;
        Ok(())
    }

    fn wait(&self) -> BackendResult<()> {
        // SAFETY: fence belongs to this device and is live.
        unsafe {
            self.device
                .device
                .wait_for_fences(&[self.fence], true, 60_000_000_000)
        }?;
        Ok(())
    }
}

impl Drop for GpuFence {
    fn drop(&mut self) {
        // SAFETY: GpuFence owns this fence.
        unsafe {
            self.device.device.destroy_fence(self.fence, None);
        }
    }
}

#[derive(Debug)]
struct GpuCommandPool {
    device: Arc<GpuDevice>,
    pool: vk::CommandPool,
}

impl GpuCommandPool {
    fn new(device: Arc<GpuDevice>) -> BackendResult<Self> {
        let create_info = vk::CommandPoolCreateInfo::default()
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )
            .queue_family_index(device.queue_family_index);
        // SAFETY: queue_family_index belongs to this device.
        let pool = unsafe { device.device.create_command_pool(&create_info, None) }?;
        Ok(Self { device, pool })
    }

    fn allocate_primary(&self) -> BackendResult<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: pool is live and belongs to this device.
        let command_buffers =
            unsafe { self.device.device.allocate_command_buffers(&allocate_info) }?;
        Ok(command_buffers[0])
    }

    fn reset(&self) -> BackendResult<()> {
        // SAFETY: callers reset only after work submitted from this pool has
        // completed, so no command buffer from the pool is in use by the GPU.
        unsafe {
            self.device
                .device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
        }?;
        Ok(())
    }
}

impl Drop for GpuCommandPool {
    fn drop(&mut self) {
        // SAFETY: GpuCommandPool owns this command pool. PendingSubmission waits
        // on its fence before dropping command pools for submitted work.
        unsafe {
            self.device.device.destroy_command_pool(self.pool, None);
        }
    }
}

#[derive(Debug)]
struct PendingSubmission {
    fence: GpuFence,
    _command_pool: GpuCommandPool,
}

impl Drop for PendingSubmission {
    fn drop(&mut self) {
        let _ = self.fence.wait();
    }
}

fn one_time_commands<T>(
    device: &Arc<GpuDevice>,
    record: impl FnOnce(vk::CommandBuffer) -> Result<T, ResourceError>,
    wait_semaphore: Option<&GpuSemaphore>,
    signal_semaphore: Option<&GpuSemaphore>,
    wait: bool,
) -> Result<(T, PendingSubmission), ResourceError> {
    let command_pool = GpuCommandPool::new(device.clone()).map_err(|err| {
        eprintln!("vulkan command pool creation failed: {err:?}");
        ResourceError::ExportFailed
    })?;
    let command_buffer = command_pool.allocate_primary().map_err(|err| {
        eprintln!("vulkan command buffer allocation failed: {err:?}");
        ResourceError::ExportFailed
    })?;
    let begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: command_buffer is newly allocated and not currently recording.
    unsafe { device.device.begin_command_buffer(command_buffer, &begin) }.map_err(|err| {
        eprintln!("vulkan command buffer begin failed: {err:?}");
        ResourceError::ExportFailed
    })?;
    let output = record(command_buffer)?;
    // SAFETY: command_buffer is currently recording.
    unsafe { device.device.end_command_buffer(command_buffer) }.map_err(|err| {
        eprintln!("vulkan command buffer end failed: {err:?}");
        ResourceError::ExportFailed
    })?;
    let fence = GpuFence::new(device.clone()).map_err(|err| {
        eprintln!("vulkan fence creation failed: {err:?}");
        ResourceError::ExportFailed
    })?;
    let wait_semaphores = wait_semaphore
        .map(|semaphore| vec![semaphore.semaphore])
        .unwrap_or_default();
    let wait_stages = wait_semaphore
        .map(|_| vec![vk::PipelineStageFlags::ALL_COMMANDS])
        .unwrap_or_default();
    let signal_semaphores = signal_semaphore
        .map(|semaphore| vec![semaphore.semaphore])
        .unwrap_or_default();
    let command_buffers = [command_buffer];
    let submit = vk::SubmitInfo::default()
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&wait_stages)
        .command_buffers(&command_buffers)
        .signal_semaphores(&signal_semaphores);
    // SAFETY: queue belongs to device, command buffer is executable, semaphore
    // and fence handles belong to same device and are kept alive by wrappers.
    unsafe {
        device
            .device
            .queue_submit(device.queue, &[submit], fence.fence)
    }
    .map_err(|err| {
        eprintln!("vulkan queue submit failed: {err:?}");
        ResourceError::ExportFailed
    })?;
    if wait {
        fence.wait().map_err(|err| {
            eprintln!("vulkan fence wait failed: {err:?}");
            ResourceError::ExportFailed
        })?;
    }
    Ok((
        output,
        PendingSubmission {
            fence,
            _command_pool: command_pool,
        },
    ))
}

fn validate_yuv422p10_upload(
    upload: &Yuv422P10Upload<'_>,
) -> Result<Yuv422P10Layout, ResourceError> {
    validate_yuv422p10_plane(upload.y, upload.width, upload.height)?;
    validate_yuv422p10_plane(upload.cb, upload.width / 2, upload.height)?;
    validate_yuv422p10_plane(upload.cr, upload.width / 2, upload.height)?;
    let cpu_surface_bytes = upload
        .y
        .data
        .len()
        .checked_add(upload.cb.data.len())
        .and_then(|total| total.checked_add(upload.cr.data.len()))
        .ok_or(ResourceError::InvalidImageDimensions)?;
    yuv422p10_layout_for_dimensions(upload.width, upload.height, cpu_surface_bytes)
}

fn validate_yuv422p10_plane(
    plane: Yuv422P10Plane<'_>,
    expected_width_samples: u32,
    expected_height: u32,
) -> Result<(), ResourceError> {
    if plane.width_samples != expected_width_samples || plane.height != expected_height {
        return Err(ResourceError::InvalidImageDimensions);
    }
    let row_bytes = row_bytes_for_u16_samples(plane.width_samples)?;
    if plane.stride_bytes < row_bytes || plane.stride_bytes == 0 {
        return Err(ResourceError::InvalidImageDimensions);
    }
    let required_len = plane
        .stride_bytes
        .checked_mul(
            usize::try_from(plane.height).map_err(|_| ResourceError::InvalidImageDimensions)?,
        )
        .ok_or(ResourceError::InvalidImageDimensions)?;
    if plane.data.len() < required_len {
        return Err(ResourceError::InvalidImageDimensions);
    }
    Ok(())
}

fn row_bytes_for_u16_samples(width_samples: u32) -> Result<usize, ResourceError> {
    usize::try_from(width_samples)
        .ok()
        .and_then(|width| width.checked_mul(2))
        .ok_or(ResourceError::InvalidImageDimensions)
}

fn checked_sample_count(width: u32, height: u32) -> Result<usize, ResourceError> {
    usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or(ResourceError::InvalidImageDimensions)
}

fn align_usize(value: usize, alignment: usize) -> Result<usize, ResourceError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(ResourceError::InvalidBufferSize);
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(ResourceError::InvalidBufferSize)
}

fn expand_yuv422p10_plane(plane: Yuv422P10Plane<'_>) -> Result<Vec<u32>, ResourceError> {
    validate_yuv422p10_plane(plane, plane.width_samples, plane.height)?;
    let width =
        usize::try_from(plane.width_samples).map_err(|_| ResourceError::InvalidImageDimensions)?;
    let height =
        usize::try_from(plane.height).map_err(|_| ResourceError::InvalidImageDimensions)?;
    let row_bytes = row_bytes_for_u16_samples(plane.width_samples)?;
    let mut values = Vec::with_capacity(
        width
            .checked_mul(height)
            .ok_or(ResourceError::InvalidImageDimensions)?,
    );
    for row in 0..height {
        let start = row
            .checked_mul(plane.stride_bytes)
            .ok_or(ResourceError::InvalidImageDimensions)?;
        let end = start
            .checked_add(row_bytes)
            .ok_or(ResourceError::InvalidImageDimensions)?;
        let row_data = plane
            .data
            .get(start..end)
            .ok_or(ResourceError::InvalidImageDimensions)?;
        let (samples, remainder) = row_data.as_chunks::<2>();
        if !remainder.is_empty() {
            return Err(ResourceError::InvalidImageDimensions);
        }
        for sample in samples {
            values.push(u32::from(u16::from_le_bytes([sample[0], sample[1]])) & 0x03ff);
        }
    }
    Ok(values)
}

fn validate_nv12_upload(upload: &Nv12Upload<'_>) -> Result<Nv12Layout, ResourceError> {
    if upload.coded_width == 0
        || upload.coded_height == 0
        || upload.visible_width == 0
        || upload.visible_height == 0
        || upload.coded_width > MAX_VIDEO_SURFACE_WIDTH
        || upload.coded_height > MAX_VIDEO_SURFACE_HEIGHT
        || upload.visible_width > upload.coded_width
        || upload.visible_height > upload.coded_height
        || !upload.coded_width.is_multiple_of(2)
        || !upload.coded_height.is_multiple_of(2)
        || !upload.visible_width.is_multiple_of(2)
    {
        return Err(ResourceError::InvalidImageDimensions);
    }
    validate_nv12_plane(upload.y, upload.coded_width, upload.coded_height)?;
    validate_nv12_plane(upload.uv, upload.coded_width, upload.coded_height / 2)?;
    let cpu_surface_bytes = upload
        .y
        .data
        .len()
        .checked_add(upload.uv.data.len())
        .ok_or(ResourceError::InvalidImageDimensions)?;
    nv12_layout_for_dimensions(
        upload.coded_width,
        upload.coded_height,
        upload.visible_width,
        upload.visible_height,
        cpu_surface_bytes,
    )
}

fn validate_nv12_plane(
    plane: Nv12Plane<'_>,
    expected_width_bytes: u32,
    expected_height: u32,
) -> Result<(), ResourceError> {
    if plane.width_bytes != expected_width_bytes || plane.height != expected_height {
        return Err(ResourceError::InvalidImageDimensions);
    }
    let row_bytes =
        usize::try_from(plane.width_bytes).map_err(|_| ResourceError::InvalidImageDimensions)?;
    if plane.stride_bytes < row_bytes || plane.stride_bytes == 0 {
        return Err(ResourceError::InvalidImageDimensions);
    }
    let required_len = plane
        .stride_bytes
        .checked_mul(
            usize::try_from(plane.height).map_err(|_| ResourceError::InvalidImageDimensions)?,
        )
        .ok_or(ResourceError::InvalidImageDimensions)?;
    if plane.data.len() < required_len {
        return Err(ResourceError::InvalidImageDimensions);
    }
    Ok(())
}

fn expand_nv12_y_plane(plane: Nv12Plane<'_>) -> Result<Vec<u32>, ResourceError> {
    let width =
        usize::try_from(plane.width_bytes).map_err(|_| ResourceError::InvalidImageDimensions)?;
    let height =
        usize::try_from(plane.height).map_err(|_| ResourceError::InvalidImageDimensions)?;
    let mut values = Vec::with_capacity(
        width
            .checked_mul(height)
            .ok_or(ResourceError::InvalidImageDimensions)?,
    );
    for row in 0..height {
        let start = row
            .checked_mul(plane.stride_bytes)
            .ok_or(ResourceError::InvalidImageDimensions)?;
        let end = start
            .checked_add(width)
            .ok_or(ResourceError::InvalidImageDimensions)?;
        let row_data = plane
            .data
            .get(start..end)
            .ok_or(ResourceError::InvalidImageDimensions)?;
        values.extend(row_data.iter().map(|value| u32::from(*value)));
    }
    Ok(values)
}

fn expand_nv12_uv_plane(plane: Nv12Plane<'_>) -> Result<Vec<u32>, ResourceError> {
    let width =
        usize::try_from(plane.width_bytes).map_err(|_| ResourceError::InvalidImageDimensions)?;
    if !width.is_multiple_of(2) {
        return Err(ResourceError::InvalidImageDimensions);
    }
    let height =
        usize::try_from(plane.height).map_err(|_| ResourceError::InvalidImageDimensions)?;
    let mut values = Vec::with_capacity(
        (width / 2)
            .checked_mul(height)
            .ok_or(ResourceError::InvalidImageDimensions)?,
    );
    for row in 0..height {
        let start = row
            .checked_mul(plane.stride_bytes)
            .ok_or(ResourceError::InvalidImageDimensions)?;
        let end = start
            .checked_add(width)
            .ok_or(ResourceError::InvalidImageDimensions)?;
        let row_data = plane
            .data
            .get(start..end)
            .ok_or(ResourceError::InvalidImageDimensions)?;
        for pair in row_data.chunks_exact(2) {
            values.push(u32::from(pair[0]) | (u32::from(pair[1]) << 8));
        }
    }
    Ok(values)
}

fn nv12_layout_for_dimensions(
    coded_width: u32,
    coded_height: u32,
    visible_width: u32,
    visible_height: u32,
    cpu_surface_bytes: usize,
) -> Result<Nv12Layout, ResourceError> {
    if coded_width == 0
        || coded_height == 0
        || visible_width == 0
        || visible_height == 0
        || visible_width > coded_width
        || visible_height > coded_height
        || !coded_width.is_multiple_of(2)
        || !coded_height.is_multiple_of(2)
        || !visible_width.is_multiple_of(2)
    {
        return Err(ResourceError::InvalidImageDimensions);
    }
    let coded_width =
        usize::try_from(coded_width).map_err(|_| ResourceError::InvalidImageDimensions)?;
    let coded_height =
        usize::try_from(coded_height).map_err(|_| ResourceError::InvalidImageDimensions)?;
    let y_stride_bytes = align_usize(coded_width, 4)?;
    let uv_stride_bytes = align_usize(coded_width, 4)?;
    let y_bytes = y_stride_bytes
        .checked_mul(coded_height)
        .ok_or(ResourceError::InvalidImageDimensions)?;
    let uv_bytes = uv_stride_bytes
        .checked_mul(coded_height / 2)
        .ok_or(ResourceError::InvalidImageDimensions)?;
    let output_pixels = checked_sample_count(visible_width, visible_height)?;
    let output_samples = output_pixels
        .checked_mul(YUV422P10_OUTPUT_COMPONENTS)
        .ok_or(ResourceError::InvalidImageDimensions)?;
    let staging_bytes = y_bytes
        .checked_add(uv_bytes)
        .ok_or(ResourceError::InvalidImageDimensions)?;
    let output_bytes = output_samples
        .checked_mul(YUV422P10_OUTPUT_BYTES_PER_COMPONENT)
        .ok_or(ResourceError::InvalidImageDimensions)?;
    Ok(Nv12Layout {
        y_bytes,
        uv_bytes,
        y_stride_bytes,
        uv_stride_bytes,
        output_samples,
        cpu_surface_bytes,
        staging_bytes,
        gpu_plane_bytes: staging_bytes,
        output_bytes,
    })
}

pub fn nv12_reference_rgba_u16(upload: &Nv12Upload<'_>) -> Result<Vec<u16>, ResourceError> {
    let layout = validate_nv12_upload(upload)?;
    let y_values = expand_nv12_y_plane(upload.y)?;
    let uv_values = expand_nv12_uv_plane(upload.uv)?;
    let coded_width =
        usize::try_from(upload.coded_width).map_err(|_| ResourceError::InvalidImageDimensions)?;
    let uv_width = coded_width / 2;
    let visible_width =
        usize::try_from(upload.visible_width).map_err(|_| ResourceError::InvalidImageDimensions)?;
    let visible_height = usize::try_from(upload.visible_height)
        .map_err(|_| ResourceError::InvalidImageDimensions)?;
    let mut out = Vec::with_capacity(layout.output_samples);
    for row in 0..visible_height {
        for x in 0..visible_width {
            let y = y_values[row * coded_width + x];
            let uv = uv_values[(row / 2) * uv_width + (x / 2)];
            let cb = uv & 0xff;
            let cr = (uv >> 8) & 0xff;
            out.extend_from_slice(&convert_rec709_limited8_to_rgba_u16(y, cb, cr));
        }
    }
    Ok(out)
}

pub fn yuv422p10_reference_rgba_u16(
    upload: &Yuv422P10Upload<'_>,
) -> Result<Vec<u16>, ResourceError> {
    let layout = validate_yuv422p10_upload(upload)?;
    let y_values = expand_yuv422p10_plane(upload.y)?;
    let cb_values = expand_yuv422p10_plane(upload.cb)?;
    let cr_values = expand_yuv422p10_plane(upload.cr)?;
    let width = usize::try_from(upload.width).map_err(|_| ResourceError::InvalidImageDimensions)?;
    let height =
        usize::try_from(upload.height).map_err(|_| ResourceError::InvalidImageDimensions)?;
    let chroma_width = width / 2;
    let mut out = Vec::with_capacity(layout.output_samples);
    for row in 0..height {
        for x in 0..width {
            let y = y_values[row * width + x];
            let chroma_index = row * chroma_width + (x / 2);
            let cb = cb_values[chroma_index];
            let cr = cr_values[chroma_index];
            out.extend_from_slice(&convert_rec709_limited_to_rgba_u16(y, cb, cr));
        }
    }
    Ok(out)
}

fn convert_rec709_limited_to_rgba_u16(y: u32, cb: u32, cr: u32) -> [u16; 4] {
    let y_limited = ((y as f32 - 64.0).max(0.0)) / 876.0;
    let cb_centered = (cb as f32 - 512.0) / 896.0;
    let cr_centered = (cr as f32 - 512.0) / 896.0;
    let r = (y_limited + 1.5748 * cr_centered).clamp(0.0, 1.0);
    let g = (y_limited - 0.187324 * cb_centered - 0.468124 * cr_centered).clamp(0.0, 1.0);
    let b = (y_limited + 1.8556 * cb_centered).clamp(0.0, 1.0);
    [
        (r * 65535.0).round() as u16,
        (g * 65535.0).round() as u16,
        (b * 65535.0).round() as u16,
        u16::MAX,
    ]
}

fn convert_rec709_limited8_to_rgba_u16(y: u32, cb: u32, cr: u32) -> [u16; 4] {
    let y_limited = ((y as f32 - 16.0).max(0.0)) / 219.0;
    let cb_centered = (cb as f32 - 128.0) / 224.0;
    let cr_centered = (cr as f32 - 128.0) / 224.0;
    let r = (y_limited + 1.5748 * cr_centered).clamp(0.0, 1.0);
    let g = (y_limited - 0.187324 * cb_centered - 0.468124 * cr_centered).clamp(0.0, 1.0);
    let b = (y_limited + 1.8556 * cb_centered).clamp(0.0, 1.0);
    [
        (r * 65535.0).round() as u16,
        (g * 65535.0).round() as u16,
        (b * 65535.0).round() as u16,
        u16::MAX,
    ]
}

pub fn yuv422p10_rgba_u16_checksum(values: &[u16]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn describe_device(
    raw_id: u64,
    properties: vk::PhysicalDeviceProperties,
) -> Result<DeviceDesc, DeviceDiscoveryError> {
    Ok(DeviceDesc {
        id: DeviceId::new(raw_id)?,
        class: classify_device_type(properties.device_type),
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
        name: bounded_name(properties.device_name.as_ptr())?,
        backend: BackendApi::Vulkan,
        api_version: to_qgs_api_version(properties.api_version),
        driver_version: properties.driver_version,
    })
}

fn describe_capabilities(device_id: DeviceId, device: &GpuDevice) -> DeviceCapabilities {
    DeviceCapabilities {
        device_id,
        compute: describe_compute_capabilities(device),
        memory: describe_memory_capabilities(device),
        interop: describe_interop_capabilities(device),
    }
}

fn describe_compute_capabilities(device: &GpuDevice) -> ComputeCapabilities {
    let limits = device.properties.limits;
    ComputeCapabilities {
        supported: device.queue_supports_compute(),
        max_workgroup_count: limits.max_compute_work_group_count,
        max_workgroup_size: limits.max_compute_work_group_size,
        max_workgroup_invocations: limits.max_compute_work_group_invocations,
    }
}

fn describe_memory_capabilities(device: &GpuDevice) -> MemoryCapabilities {
    let mut heaps = Vec::new();
    for index in 0..device.memory_properties.memory_heap_count {
        if heaps.len() >= MAX_MEMORY_HEAP_COUNT {
            break;
        }
        let heap = device.memory_properties.memory_heaps[index as usize];
        heaps.push(MemoryHeapDesc {
            size_bytes: heap.size,
            device_local: heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
        });
    }

    let mut host_visible = false;
    let mut host_coherent = false;
    let mut device_local = false;
    for index in 0..device
        .memory_properties
        .memory_type_count
        .min(MAX_MEMORY_TYPE_COUNT as u32)
    {
        let flags = device.memory_properties.memory_types[index as usize].property_flags;
        host_visible |= flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
        host_coherent |= flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
        device_local |= flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
    }

    MemoryCapabilities {
        heaps,
        memory_type_count: device.memory_properties.memory_type_count as u16,
        host_visible,
        host_coherent,
        device_local,
    }
}

fn describe_interop_capabilities(device: &GpuDevice) -> InteropCapabilities {
    InteropCapabilities {
        external_memory_fd: device.extension_support.khr_external_memory
            && device.extension_support.khr_external_memory_fd,
        dma_buf: device.extension_support.khr_external_memory
            && device.extension_support.khr_external_memory_fd
            && device.extension_support.ext_external_memory_dma_buf,
        external_semaphore_fd: device.extension_support.khr_external_semaphore
            && device.extension_support.khr_external_semaphore_fd,
        external_fence_fd: device.extension_support.khr_external_fence_fd,
    }
}

fn classify_device_type(device_type: vk::PhysicalDeviceType) -> DeviceClass {
    match device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => DeviceClass::DiscreteGpu,
        vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceClass::IntegratedGpu,
        vk::PhysicalDeviceType::CPU => DeviceClass::Software,
        vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceClass::Other,
        _ => DeviceClass::Other,
    }
}

fn bounded_name(ptr: *const std::os::raw::c_char) -> Result<String, DeviceDiscoveryError> {
    // SAFETY: Vulkan guarantees device_name is a null-terminated fixed-size C string.
    let name = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|_| DeviceDiscoveryError::BackendFailed)?;
    let mut name = name.to_string();
    if name.len() > MAX_DEVICE_NAME_LEN {
        name.truncate(MAX_DEVICE_NAME_LEN);
    }
    Ok(name)
}

fn to_qgs_api_version(version: u32) -> ApiVersion {
    ApiVersion {
        major: vk::api_version_major(version) as u16,
        minor: vk::api_version_minor(version) as u16,
        patch: vk::api_version_patch(version) as u16,
    }
}

fn select_queue_family(queues: &[vk::QueueFamilyProperties]) -> Option<u32> {
    queues
        .iter()
        .enumerate()
        .find(|(_, queue)| {
            queue.queue_count > 0 && queue.queue_flags.contains(vk::QueueFlags::COMPUTE)
        })
        .or_else(|| {
            queues
                .iter()
                .enumerate()
                .find(|(_, queue)| queue.queue_count > 0)
        })
        .and_then(|(index, _)| u32::try_from(index).ok())
}

fn map_buffer_usage(usage: BufferUsageFlags) -> Result<vk::BufferUsageFlags, ResourceError> {
    let mut flags = vk::BufferUsageFlags::empty();
    if usage.contains(BufferUsageFlags::TRANSFER_SRC) {
        flags |= vk::BufferUsageFlags::TRANSFER_SRC;
    }
    if usage.contains(BufferUsageFlags::TRANSFER_DST) {
        flags |= vk::BufferUsageFlags::TRANSFER_DST;
    }
    if usage.contains(BufferUsageFlags::STORAGE) {
        flags |= vk::BufferUsageFlags::STORAGE_BUFFER;
    }
    if flags.is_empty() {
        Err(ResourceError::UnsupportedMemoryRequirements)
    } else {
        Ok(flags)
    }
}

fn map_pixel_format(format: PixelFormat) -> Result<vk::Format, ResourceError> {
    match format {
        PixelFormat::Rgba8Unorm => Ok(vk::Format::R8G8B8A8_UNORM),
    }
}

fn map_image_usage(usage: ImageUsageFlags) -> Result<vk::ImageUsageFlags, ResourceError> {
    let mut flags = vk::ImageUsageFlags::empty();
    if usage.contains(ImageUsageFlags::TRANSFER_SRC) {
        flags |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    if usage.contains(ImageUsageFlags::TRANSFER_DST) {
        flags |= vk::ImageUsageFlags::TRANSFER_DST;
    }
    if usage.contains(ImageUsageFlags::STORAGE) {
        flags |= vk::ImageUsageFlags::STORAGE;
    }
    if flags.is_empty() {
        Err(ResourceError::UnsupportedImageUsage)
    } else {
        Ok(flags)
    }
}

fn required_memory_flags(desc: &BufferDesc) -> vk::MemoryPropertyFlags {
    let mut flags = vk::MemoryPropertyFlags::empty();
    if desc.memory_preference.host_visible_required {
        flags |= vk::MemoryPropertyFlags::HOST_VISIBLE;
    }
    flags
}

fn preferred_memory_flags(desc: &BufferDesc) -> vk::MemoryPropertyFlags {
    let mut flags = vk::MemoryPropertyFlags::empty();
    if desc.memory_preference.device_preferred {
        flags |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
    }
    if desc.memory_preference.host_coherent_preferred {
        flags |= vk::MemoryPropertyFlags::HOST_COHERENT;
    }
    flags
}

fn selected_memory_properties(
    device: &GpuDevice,
    memory_type_index: u32,
) -> Result<SelectedMemoryProperties, ResourceError> {
    let memory_type = device
        .memory_properties
        .memory_types
        .get(memory_type_index as usize)
        .ok_or(ResourceError::UnsupportedMemoryRequirements)?;
    let flags = memory_type.property_flags;
    Ok(SelectedMemoryProperties {
        device_local: flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL),
        host_visible: flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
        host_coherent: flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT),
    })
}

fn map_external_memory_handle_type(
    handle_type: ExternalHandleType,
) -> Result<vk::ExternalMemoryHandleTypeFlags, ResourceError> {
    match handle_type {
        ExternalHandleType::OpaqueFd => Ok(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD),
        ExternalHandleType::DmaBuf => Ok(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT),
    }
}

fn validate_external_buffer_support(
    device: &GpuDevice,
    usage: vk::BufferUsageFlags,
    handle_type: vk::ExternalMemoryHandleTypeFlags,
) -> Result<(), ResourceError> {
    if handle_type == vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT
        && !device.extension_support.ext_external_memory_dma_buf
    {
        return Err(ResourceError::UnsupportedExternalHandleType);
    }
    if !device.extension_support.khr_external_memory
        || !device.extension_support.khr_external_memory_fd
    {
        return Err(ResourceError::UnsupportedExternalHandleType);
    }
    let info = vk::PhysicalDeviceExternalBufferInfo::default()
        .flags(vk::BufferCreateFlags::empty())
        .usage(usage)
        .handle_type(handle_type);
    let mut properties = vk::ExternalBufferProperties::default();
    // SAFETY: device.physical_device belongs to device.instance, and info is valid.
    unsafe {
        device
            .instance
            .instance
            .get_physical_device_external_buffer_properties(
                device.physical_device,
                &info,
                &mut properties,
            )
    };
    let features = properties
        .external_memory_properties
        .external_memory_features;
    if features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
        && features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
    {
        Ok(())
    } else {
        Err(ResourceError::UnsupportedExternalHandleType)
    }
}

fn validate_external_image_support(
    device: &GpuDevice,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    tiling: vk::ImageTiling,
    handle_type: vk::ExternalMemoryHandleTypeFlags,
) -> Result<(), ResourceError> {
    if handle_type == vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT
        && !device.extension_support.ext_external_memory_dma_buf
    {
        return Err(ResourceError::UnsupportedExternalHandleType);
    }
    if !device.extension_support.khr_external_memory
        || !device.extension_support.khr_external_memory_fd
    {
        return Err(ResourceError::UnsupportedExternalHandleType);
    }
    let mut info = vk::PhysicalDeviceExternalImageFormatInfo::default().handle_type(handle_type);
    let mut external_properties = vk::ExternalImageFormatProperties::default();
    let mut format_properties =
        vk::ImageFormatProperties2::default().push_next(&mut external_properties);
    let format_info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(format)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(tiling)
        .usage(usage)
        .flags(vk::ImageCreateFlags::empty())
        .push_next(&mut info);
    // SAFETY: physical device belongs to instance, and pNext structures live
    // through the call.
    unsafe {
        device
            .instance
            .instance
            .get_physical_device_image_format_properties2(
                device.physical_device,
                &format_info,
                &mut format_properties,
            )
    }
    .map_err(|_| ResourceError::UnsupportedExternalHandleType)?;
    let features = external_properties
        .external_memory_properties
        .external_memory_features;
    if features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
        && features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
    {
        Ok(())
    } else {
        Err(ResourceError::UnsupportedExternalHandleType)
    }
}

fn validate_sync_fd_support(device: &GpuDevice) -> Result<(), SyncError> {
    if device.extension_support.khr_external_semaphore
        && device.extension_support.khr_external_semaphore_fd
    {
        Ok(())
    } else {
        Err(SyncError::UnsupportedSyncHandleType)
    }
}

fn import_external_buffer(
    device: &Arc<GpuDevice>,
    metadata: &ExportedResourceMetadata,
    usage: vk::BufferUsageFlags,
    handle_type: vk::ExternalMemoryHandleTypeFlags,
    handle: File,
) -> Result<GpuBuffer, ResourceError> {
    let fd_memory_type_bits = device
        .memory_fd_type_bits(handle_type, &handle)
        .map_err(|err| {
            eprintln!("vulkan imported buffer fd-property query failed: {err:?}");
            ResourceError::ExportFailed
        })?;
    let memory_type_bit = 1_u32
        .checked_shl(metadata.backend_memory_type_index)
        .ok_or(ResourceError::UnsupportedMemoryRequirements)?;
    if fd_memory_type_bits & memory_type_bit == 0 {
        return Err(ResourceError::UnsupportedMemoryRequirements);
    }
    GpuBuffer::imported(
        device.clone(),
        ImportedBufferInfo {
            size: metadata.size_bytes,
            usage,
            handle_type,
            file: handle,
            allocation_size: metadata.allocation_size_bytes,
            memory_type_index: metadata.backend_memory_type_index,
            dedicated: metadata.dedicated_allocation,
        },
    )
}

struct ImportedImageInfo<'a> {
    metadata: &'a ExportedResourceMetadata,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    tiling: vk::ImageTiling,
    handle_type: vk::ExternalMemoryHandleTypeFlags,
    handle: File,
}

fn import_external_image(
    device: &Arc<GpuDevice>,
    info: ImportedImageInfo<'_>,
) -> Result<GpuImage, ResourceError> {
    let fd_memory_type_bits = device
        .memory_fd_type_bits(info.handle_type, &info.handle)
        .map_err(|err| {
            eprintln!("vulkan imported image fd-property query failed: {err:?}");
            ResourceError::ExportFailed
        })?;
    let memory_type_bit = 1_u32
        .checked_shl(info.metadata.backend_memory_type_index)
        .ok_or(ResourceError::UnsupportedMemoryRequirements)?;
    if fd_memory_type_bits & memory_type_bit == 0 {
        return Err(ResourceError::UnsupportedMemoryRequirements);
    }
    GpuImage::imported(
        device.clone(),
        info.metadata,
        info.format,
        info.usage,
        info.tiling,
        info.handle_type,
        info.handle,
    )
}

fn submit_gpu_fill(
    buffer: &GpuBuffer,
    semaphore: &GpuSemaphore,
    pattern: u32,
) -> Result<PendingSubmission, SyncError> {
    let (_, pending) = one_time_commands(
        &buffer.device,
        |command_buffer| {
            // SAFETY: command_buffer is recording. buffer is live and created
            // with TRANSFER_DST usage by QGS callers that request sync export.
            unsafe {
                buffer.device.device.cmd_fill_buffer(
                    command_buffer,
                    buffer.buffer,
                    0,
                    buffer.size,
                    pattern,
                );
            }
            Ok(())
        },
        None,
        Some(semaphore),
        false,
    )
    .map_err(|_| SyncError::SyncExportFailed)?;
    Ok(pending)
}

fn submit_gpu_image_clear(
    image: &GpuImage,
    semaphore: &GpuSemaphore,
    pattern: u32,
) -> Result<PendingSubmission, SyncError> {
    let (_, pending) = one_time_commands(
        &image.device,
        |command_buffer| {
            transition_image(
                &image.device,
                command_buffer,
                ImageTransition {
                    image: image.image,
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::GENERAL,
                    src_access: vk::AccessFlags::empty(),
                    dst_access: vk::AccessFlags::TRANSFER_WRITE,
                    src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                    dst_stage: vk::PipelineStageFlags::TRANSFER,
                },
            );
            let bytes = pattern.to_le_bytes();
            let color = vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, f32::from(bytes[3]) / f32::from(u8::MAX)],
            };
            // SAFETY: command_buffer is recording and image has TRANSFER_DST
            // usage. QGS transitioned it to GENERAL immediately above.
            unsafe {
                image.device.device.cmd_clear_color_image(
                    command_buffer,
                    image.image,
                    vk::ImageLayout::GENERAL,
                    &color,
                    &[color_subresource_range()],
                );
            }
            Ok(())
        },
        None,
        Some(semaphore),
        false,
    )
    .map_err(|_| SyncError::SyncExportFailed)?;
    Ok(pending)
}

fn validate_synced_gpu_copy(
    device: &Arc<GpuDevice>,
    imported_buffer: &GpuBuffer,
    semaphore: &GpuSemaphore,
    pattern: u32,
) -> Result<(), ResourceError> {
    let readback = GpuBuffer::new(
        device.clone(),
        std::mem::size_of::<u32>() as u64,
        vk::BufferUsageFlags::TRANSFER_DST,
        None,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
        vk::MemoryPropertyFlags::HOST_COHERENT,
        false,
    )?;
    let (_, _pending) = one_time_commands(
        device,
        |command_buffer| {
            let copy = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(std::mem::size_of::<u32>() as u64);
            // SAFETY: command buffer is recording and both buffers are live with
            // the required transfer usages.
            unsafe {
                device.device.cmd_copy_buffer(
                    command_buffer,
                    imported_buffer.buffer,
                    readback.buffer,
                    &[copy],
                );
            }
            Ok(())
        },
        Some(semaphore),
        None,
        true,
    )?;
    let bytes = readback.read_bytes(0, std::mem::size_of::<u32>())?;
    let observed = u32::from_le_bytes(bytes.try_into().map_err(|_| ResourceError::ExportFailed)?);
    if observed == pattern {
        Ok(())
    } else {
        Err(ResourceError::ExportFailed)
    }
}

fn run_compute_increment_proof(
    device: &Arc<GpuDevice>,
    imported_buffer: &GpuBuffer,
    semaphore: &GpuSemaphore,
    input_values: &[u32],
    input_bytes: u64,
) -> Result<Vec<u32>, ResourceError> {
    let upload = GpuBuffer::new(
        device.clone(),
        input_bytes,
        vk::BufferUsageFlags::TRANSFER_SRC,
        None,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
        vk::MemoryPropertyFlags::HOST_COHERENT,
        false,
    )?;
    upload.write_bytes(0, cast_u32_slice(input_values))?;
    let readback = GpuBuffer::new(
        device.clone(),
        input_bytes,
        vk::BufferUsageFlags::TRANSFER_DST,
        None,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
        vk::MemoryPropertyFlags::HOST_COHERENT,
        false,
    )?;
    let pipeline = ComputePipelineState::new_storage_buffer(device.clone())?;
    let descriptor = pipeline.write_storage_buffer(imported_buffer)?;
    let groups = u32::try_from(input_values.len() / COMPUTE_LOCAL_SIZE_X as usize)
        .map_err(|_| ResourceError::InvalidBufferSize)?;

    let (_, _pending) = one_time_commands(
        device,
        |command_buffer| {
            let copy = vk::BufferCopy::default().size(input_bytes);
            // SAFETY: command buffer is recording; buffers and pipeline objects
            // are live for the whole submission.
            unsafe {
                device.device.cmd_copy_buffer(
                    command_buffer,
                    upload.buffer,
                    imported_buffer.buffer,
                    &[copy],
                );
                device.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.pipeline,
                );
                device.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.layout,
                    0,
                    &[descriptor.set],
                    &[],
                );
                device.device.cmd_dispatch(command_buffer, groups, 1, 1);
                device.device.cmd_copy_buffer(
                    command_buffer,
                    imported_buffer.buffer,
                    readback.buffer,
                    &[copy],
                );
            }
            Ok(())
        },
        Some(semaphore),
        None,
        true,
    )?;
    let bytes = readback.read_bytes(0, input_bytes as usize)?;
    let mut out = Vec::with_capacity(input_values.len());
    let (chunks, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(ResourceError::ExportFailed);
    }
    for chunk in chunks {
        out.push(u32::from_le_bytes(*chunk));
    }
    Ok(out)
}

fn run_image_invert_proof(
    device: &Arc<GpuDevice>,
    imported_image: &GpuImage,
    semaphore: &GpuSemaphore,
    input_pixels: &[u8],
) -> Result<Vec<u8>, ResourceError> {
    let readback = GpuBuffer::new(
        device.clone(),
        input_pixels.len() as u64,
        vk::BufferUsageFlags::TRANSFER_DST,
        None,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
        vk::MemoryPropertyFlags::HOST_COHERENT,
        false,
    )?;
    let view = imported_image.create_view().map_err(|err| {
        eprintln!("vulkan imported image view creation failed: {err:?}");
        ResourceError::ExportFailed
    })?;
    let pipeline = ComputePipelineState::new_storage_image(device.clone())?;
    let descriptor = pipeline.write_storage_image(&view)?;

    let (_, _pending) = one_time_commands(
        device,
        |command_buffer| {
            transition_image(
                device,
                command_buffer,
                ImageTransition {
                    image: imported_image.image,
                    old_layout: vk::ImageLayout::GENERAL,
                    new_layout: vk::ImageLayout::GENERAL,
                    src_access: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                    src_stage: vk::PipelineStageFlags::TRANSFER,
                    dst_stage: vk::PipelineStageFlags::COMPUTE_SHADER,
                },
            );
            // SAFETY: command buffer is recording; image/view/pipeline/buffer are
            // live and usages were validated before import.
            unsafe {
                device.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.pipeline,
                );
                device.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.layout,
                    0,
                    &[descriptor.set],
                    &[],
                );
                device.device.cmd_dispatch(
                    command_buffer,
                    imported_image.extent[0].div_ceil(8),
                    imported_image.extent[1].div_ceil(8),
                    1,
                );
            }
            transition_image(
                device,
                command_buffer,
                ImageTransition {
                    image: imported_image.image,
                    old_layout: vk::ImageLayout::GENERAL,
                    new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    src_access: vk::AccessFlags::SHADER_WRITE,
                    dst_access: vk::AccessFlags::TRANSFER_READ,
                    src_stage: vk::PipelineStageFlags::COMPUTE_SHADER,
                    dst_stage: vk::PipelineStageFlags::TRANSFER,
                },
            );
            let copy = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: imported_image.extent[0],
                    height: imported_image.extent[1],
                    depth: 1,
                });
            // SAFETY: command buffer is recording; image is in TRANSFER_SRC_OPTIMAL
            // and readback buffer has TRANSFER_DST usage.
            unsafe {
                device.device.cmd_copy_image_to_buffer(
                    command_buffer,
                    imported_image.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    readback.buffer,
                    &[copy],
                );
            }
            Ok(())
        },
        Some(semaphore),
        None,
        true,
    )?;
    readback.read_bytes(0, input_pixels.len())
}

pub struct GpuFrameProcessor {
    device: Arc<GpuDevice>,
    config: GpuFrameProcessorConfig,
    pipeline: Yuv422P10PipelineState,
    slots: Vec<FrameSlot>,
    next_token: u64,
    next_sequence: u64,
    counters: GpuFrameProcessorCounters,
}

impl GpuFrameProcessor {
    pub fn new(
        discovery: &VulkanDeviceDiscovery,
        config: GpuFrameProcessorConfig,
    ) -> Result<Self, FrameProcessorError> {
        validate_frame_processor_config(config)?;
        let device = discovery
            .devices
            .iter()
            .find(|device| device.desc.id == config.device_id)
            .ok_or(FrameProcessorError::Resource(
                ResourceError::UnknownDeviceId,
            ))?
            .device
            .clone();
        if !device.queue_supports_compute() {
            return Err(FrameProcessorError::Resource(
                ResourceError::UnsupportedMemoryRequirements,
            ));
        }
        let layout = yuv422p10_layout_for_dimensions(config.width, config.height, 0)?;
        let pipeline = Yuv422P10PipelineState::new(
            device.clone(),
            u32::try_from(config.slot_count).map_err(|_| ResourceError::InvalidBufferSize)?,
        )?;
        let mut slots = Vec::with_capacity(config.slot_count);
        for index in 0..config.slot_count {
            slots.push(FrameSlot::new(index, device.clone(), &pipeline, layout)?);
        }
        let slot_count = config.slot_count as u64;
        Ok(Self {
            device,
            config,
            pipeline,
            slots,
            next_token: 1,
            next_sequence: 1,
            counters: GpuFrameProcessorCounters {
                pipeline_creations: 1,
                shader_module_creations: 1,
                staging_allocations: slot_count * 3,
                gpu_plane_allocations: slot_count * 3,
                output_allocations: slot_count,
                readback_allocations: slot_count,
                command_buffer_count: slot_count,
                ..Default::default()
            },
        })
    }

    pub fn submit_frame(
        &mut self,
        upload: &Yuv422P10Upload<'_>,
        identity: FrameIdentity,
    ) -> Result<FrameToken, FrameProcessorError> {
        if upload.device_id != self.config.device_id
            || upload.width != self.config.width
            || upload.height != self.config.height
            || upload.conversion != self.config.conversion
        {
            return Err(FrameProcessorError::Resource(
                ResourceError::InvalidImageDimensions,
            ));
        }
        let layout = validate_yuv422p10_upload(upload)?;
        let slot = self
            .slots
            .iter_mut()
            .find(|slot| slot.state.can_submit())
            .ok_or(FrameProcessorError::NoFrameSlotAvailable)?;
        let token = FrameToken(self.next_token);
        self.next_token = self
            .next_token
            .checked_add(1)
            .ok_or(FrameProcessorError::InvalidFrameState)?;
        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(FrameProcessorError::InvalidFrameState)?;
        if slot.had_submission {
            self.counters.slot_reuses = self.counters.slot_reuses.saturating_add(1);
        }
        slot.submit(
            &self.device,
            &self.pipeline,
            FrameSubmission {
                upload,
                layout,
                token,
                identity,
                sequence,
            },
        )?;
        self.counters.frame_submissions = self.counters.frame_submissions.saturating_add(1);
        Ok(token)
    }

    pub fn poll_completed(&mut self) -> Result<Vec<FrameToken>, FrameProcessorError> {
        let mut completed = Vec::new();
        for slot in &mut self.slots {
            if slot.state == FrameSlotState::Submitted
                && slot.fence.is_signaled().map_err(|err| {
                    eprintln!("vulkan frame fence status query failed: {err:?}");
                    FrameProcessorError::Resource(ResourceError::ExportFailed)
                })?
            {
                slot.state = FrameSlotState::Completed;
                if let Some(info) = slot.submitted {
                    completed.push(info.token);
                }
            }
        }
        Ok(completed)
    }

    pub fn retire_completed(&mut self) -> Result<usize, FrameProcessorError> {
        self.poll_completed()?;
        let mut retired = 0;
        for slot in &mut self.slots {
            if slot.state.can_retire() {
                slot.submitted = None;
                slot.state = FrameSlotState::Available;
                retired += 1;
            }
        }
        Ok(retired)
    }

    pub fn retire_completed_token(&mut self, token: FrameToken) -> Result<(), FrameProcessorError> {
        self.poll_completed()?;
        let slot = self
            .slots
            .iter_mut()
            .find(|slot| slot.submitted.is_some_and(|info| info.token == token))
            .ok_or(FrameProcessorError::UnknownFrameToken)?;
        if !slot.state.can_retire() {
            return Err(FrameProcessorError::FrameNotComplete);
        }
        slot.submitted = None;
        slot.state = FrameSlotState::Available;
        Ok(())
    }

    pub fn wait_for_frame(
        &mut self,
        token: FrameToken,
    ) -> Result<ProcessedFrameOutput, FrameProcessorError> {
        let slot = self
            .slots
            .iter_mut()
            .find(|slot| slot.submitted.is_some_and(|info| info.token == token))
            .ok_or(FrameProcessorError::UnknownFrameToken)?;
        match slot.state {
            FrameSlotState::Submitted => {
                slot.fence.wait().map_err(|err| {
                    eprintln!("vulkan frame fence wait failed: {err:?}");
                    FrameProcessorError::Resource(ResourceError::ExportFailed)
                })?;
                slot.state = FrameSlotState::Completed;
            }
            FrameSlotState::Completed => {}
            _ => return Err(FrameProcessorError::InvalidFrameState),
        }
        let output = slot.read_completed_output()?;
        slot.submitted = None;
        slot.state = FrameSlotState::Available;
        Ok(output)
    }

    pub fn counters(&self) -> GpuFrameProcessorCounters {
        self.counters
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }
}

impl Drop for GpuFrameProcessor {
    fn drop(&mut self) {
        for slot in &mut self.slots {
            if slot.state == FrameSlotState::Submitted {
                let _ = slot.fence.wait();
                slot.state = FrameSlotState::Completed;
            }
        }
    }
}

pub struct Nv12FrameProcessor {
    device: Arc<GpuDevice>,
    config: Nv12FrameProcessorConfig,
    pipeline: Nv12PipelineState,
    slots: Vec<Nv12FrameSlot>,
    next_token: u64,
    next_sequence: u64,
    counters: GpuFrameProcessorCounters,
    diagnostics: Nv12FrameProcessorDiagnostics,
}

impl Nv12FrameProcessor {
    pub fn new(
        discovery: &VulkanDeviceDiscovery,
        config: Nv12FrameProcessorConfig,
    ) -> Result<Self, FrameProcessorError> {
        validate_nv12_frame_processor_config(config)?;
        let device = discovery
            .devices
            .iter()
            .find(|device| device.desc.id == config.device_id)
            .ok_or(FrameProcessorError::Resource(
                ResourceError::UnknownDeviceId,
            ))?
            .device
            .clone();
        if !device.queue_supports_compute() {
            return Err(FrameProcessorError::Resource(
                ResourceError::UnsupportedMemoryRequirements,
            ));
        }
        let layout = nv12_layout_for_dimensions(
            config.coded_width,
            config.coded_height,
            config.visible_width,
            config.visible_height,
            0,
        )?;
        let pipeline = Nv12PipelineState::new(
            device.clone(),
            u32::try_from(config.slot_count).map_err(|_| ResourceError::InvalidBufferSize)?,
        )?;
        let mut slots = Vec::with_capacity(config.slot_count);
        for index in 0..config.slot_count {
            slots.push(Nv12FrameSlot::new(
                index,
                device.clone(),
                &pipeline,
                layout,
            )?);
        }
        let slot_count = config.slot_count as u64;
        Ok(Self {
            device,
            config,
            pipeline,
            slots,
            next_token: 1,
            next_sequence: 1,
            counters: GpuFrameProcessorCounters {
                pipeline_creations: 1,
                shader_module_creations: 1,
                staging_allocations: slot_count * 2,
                gpu_plane_allocations: slot_count * 2,
                output_allocations: slot_count,
                readback_allocations: slot_count,
                command_buffer_count: slot_count,
                ..Default::default()
            },
            diagnostics: Nv12FrameProcessorDiagnostics {
                output_readback_enabled: config.validation_readback,
                ..Default::default()
            },
        })
    }

    pub fn submit_frame(
        &mut self,
        upload: &Nv12Upload<'_>,
        identity: FrameIdentity,
    ) -> Result<FrameToken, FrameProcessorError> {
        if upload.device_id != self.config.device_id
            || upload.coded_width != self.config.coded_width
            || upload.coded_height != self.config.coded_height
            || upload.visible_width != self.config.visible_width
            || upload.visible_height != self.config.visible_height
            || upload.conversion != self.config.conversion
        {
            return Err(FrameProcessorError::Resource(
                ResourceError::InvalidImageDimensions,
            ));
        }
        let layout = validate_nv12_upload(upload)?;
        let slot_started = Instant::now();
        let slot = self
            .slots
            .iter_mut()
            .find(|slot| slot.state.can_submit())
            .ok_or(FrameProcessorError::NoFrameSlotAvailable)?;
        self.diagnostics.slot_acquire_ns = self
            .diagnostics
            .slot_acquire_ns
            .saturating_add(slot_started.elapsed().as_nanos());
        let token = FrameToken(self.next_token);
        self.next_token = self
            .next_token
            .checked_add(1)
            .ok_or(FrameProcessorError::InvalidFrameState)?;
        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(FrameProcessorError::InvalidFrameState)?;
        if slot.had_submission {
            self.counters.slot_reuses = self.counters.slot_reuses.saturating_add(1);
        }
        slot.submit(
            &self.device,
            &self.pipeline,
            Nv12FrameSubmission {
                upload,
                layout,
                token,
                identity,
                sequence,
                validation_readback: self.config.validation_readback,
            },
            &mut self.diagnostics,
        )?;
        self.diagnostics.submit_calls = self.diagnostics.submit_calls.saturating_add(1);
        self.counters.frame_submissions = self.counters.frame_submissions.saturating_add(1);
        Ok(token)
    }

    pub fn poll_completed(&mut self) -> Result<Vec<FrameToken>, FrameProcessorError> {
        let started = Instant::now();
        self.diagnostics.poll_calls = self.diagnostics.poll_calls.saturating_add(1);
        let mut completed = Vec::new();
        for slot in &mut self.slots {
            if slot.state == FrameSlotState::Submitted
                && slot.fence.is_signaled().map_err(|err| {
                    eprintln!("vulkan NV12 frame fence status query failed: {err:?}");
                    FrameProcessorError::Resource(ResourceError::ExportFailed)
                })?
            {
                slot.state = FrameSlotState::Completed;
                if let Some(info) = slot.submitted {
                    completed.push(info.token);
                }
            }
        }
        self.diagnostics.poll_ns = self
            .diagnostics
            .poll_ns
            .saturating_add(started.elapsed().as_nanos());
        Ok(completed)
    }

    pub fn retire_completed(&mut self) -> Result<usize, FrameProcessorError> {
        self.poll_completed()?;
        let mut retired = 0;
        for slot in &mut self.slots {
            if slot.state.can_retire() {
                slot.submitted = None;
                slot.state = FrameSlotState::Available;
                retired += 1;
            }
        }
        Ok(retired)
    }

    pub fn retire_completed_token(&mut self, token: FrameToken) -> Result<(), FrameProcessorError> {
        let started = Instant::now();
        self.poll_completed()?;
        let slot = self
            .slots
            .iter_mut()
            .find(|slot| slot.submitted.is_some_and(|info| info.token == token))
            .ok_or(FrameProcessorError::UnknownFrameToken)?;
        if !slot.state.can_retire() {
            return Err(FrameProcessorError::FrameNotComplete);
        }
        slot.submitted = None;
        slot.state = FrameSlotState::Available;
        self.diagnostics.retire_calls = self.diagnostics.retire_calls.saturating_add(1);
        self.diagnostics.retire_ns = self
            .diagnostics
            .retire_ns
            .saturating_add(started.elapsed().as_nanos());
        Ok(())
    }

    pub fn wait_for_frame(
        &mut self,
        token: FrameToken,
    ) -> Result<ProcessedFrameOutput, FrameProcessorError> {
        let slot = self
            .slots
            .iter_mut()
            .find(|slot| slot.submitted.is_some_and(|info| info.token == token))
            .ok_or(FrameProcessorError::UnknownFrameToken)?;
        match slot.state {
            FrameSlotState::Submitted => {
                slot.fence.wait().map_err(|err| {
                    eprintln!("vulkan NV12 frame fence wait failed: {err:?}");
                    FrameProcessorError::Resource(ResourceError::ExportFailed)
                })?;
                slot.state = FrameSlotState::Completed;
            }
            FrameSlotState::Completed => {}
            _ => return Err(FrameProcessorError::InvalidFrameState),
        }
        let output = slot.read_completed_output()?;
        slot.submitted = None;
        slot.state = FrameSlotState::Available;
        Ok(output)
    }

    pub fn wait_for_completion(&mut self, token: FrameToken) -> Result<(), FrameProcessorError> {
        let slot = self
            .slots
            .iter_mut()
            .find(|slot| slot.submitted.is_some_and(|info| info.token == token))
            .ok_or(FrameProcessorError::UnknownFrameToken)?;
        match slot.state {
            FrameSlotState::Submitted => {
                let started = Instant::now();
                slot.fence.wait().map_err(|err| {
                    eprintln!("vulkan NV12 frame fence wait failed: {err:?}");
                    FrameProcessorError::Resource(ResourceError::ExportFailed)
                })?;
                self.diagnostics.fence_wait_calls =
                    self.diagnostics.fence_wait_calls.saturating_add(1);
                self.diagnostics.fence_wait_ns = self
                    .diagnostics
                    .fence_wait_ns
                    .saturating_add(started.elapsed().as_nanos());
                slot.state = FrameSlotState::Completed;
            }
            FrameSlotState::Completed => {}
            _ => return Err(FrameProcessorError::InvalidFrameState),
        }
        Ok(())
    }

    pub fn counters(&self) -> GpuFrameProcessorCounters {
        self.counters
    }

    pub fn diagnostics(&self) -> Nv12FrameProcessorDiagnostics {
        self.diagnostics.clone()
    }
}

impl Drop for Nv12FrameProcessor {
    fn drop(&mut self) {
        for slot in &mut self.slots {
            if slot.state == FrameSlotState::Submitted {
                let _ = slot.fence.wait();
                slot.state = FrameSlotState::Completed;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FrameSlotState {
    Available,
    Preparing,
    Submitted,
    Completed,
}

impl FrameSlotState {
    fn can_submit(self) -> bool {
        self == Self::Available
    }

    fn can_retire(self) -> bool {
        self == Self::Completed
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SubmittedFrameInfo {
    token: FrameToken,
    identity: FrameIdentity,
    sequence: u64,
    width: u32,
    height: u32,
    layout: Yuv422P10Layout,
}

#[derive(Clone, Copy)]
struct FrameSubmission<'a> {
    upload: &'a Yuv422P10Upload<'a>,
    layout: Yuv422P10Layout,
    token: FrameToken,
    identity: FrameIdentity,
    sequence: u64,
}

struct FrameSlot {
    index: usize,
    state: FrameSlotState,
    submitted: Option<SubmittedFrameInfo>,
    had_submission: bool,
    staging_y: GpuBuffer,
    staging_cb: GpuBuffer,
    staging_cr: GpuBuffer,
    gpu_y: GpuBuffer,
    gpu_cb: GpuBuffer,
    gpu_cr: GpuBuffer,
    output: GpuBuffer,
    readback: GpuBuffer,
    command_pool: GpuCommandPool,
    command_buffer: vk::CommandBuffer,
    fence: GpuFence,
    descriptor: DescriptorSet,
}

#[derive(Clone, Copy)]
struct Nv12SubmittedFrameInfo {
    token: FrameToken,
    identity: FrameIdentity,
    sequence: u64,
    width: u32,
    height: u32,
    layout: Nv12Layout,
}

#[derive(Clone, Copy)]
struct Nv12FrameSubmission<'a> {
    upload: &'a Nv12Upload<'a>,
    layout: Nv12Layout,
    token: FrameToken,
    identity: FrameIdentity,
    sequence: u64,
    validation_readback: bool,
}

struct Nv12FrameSlot {
    index: usize,
    state: FrameSlotState,
    submitted: Option<Nv12SubmittedFrameInfo>,
    had_submission: bool,
    staging_y: GpuBuffer,
    staging_uv: GpuBuffer,
    gpu_y: GpuBuffer,
    gpu_uv: GpuBuffer,
    output: GpuBuffer,
    readback: GpuBuffer,
    command_pool: GpuCommandPool,
    command_buffer: vk::CommandBuffer,
    fence: GpuFence,
    descriptor: DescriptorSet,
}

impl Nv12FrameSlot {
    fn new(
        index: usize,
        device: Arc<GpuDevice>,
        pipeline: &Nv12PipelineState,
        layout: Nv12Layout,
    ) -> Result<Self, FrameProcessorError> {
        let y_bytes =
            u64::try_from(layout.y_bytes).map_err(|_| ResourceError::InvalidBufferSize)?;
        let uv_bytes =
            u64::try_from(layout.uv_bytes).map_err(|_| ResourceError::InvalidBufferSize)?;
        let output_u32_bytes = checked_byte_len(layout.output_samples, std::mem::size_of::<u32>())?;
        let staging_y = frame_staging_buffer(&device, y_bytes)?;
        let staging_uv = frame_staging_buffer(&device, uv_bytes)?;
        let gpu_y = frame_gpu_plane_buffer(&device, y_bytes)?;
        let gpu_uv = frame_gpu_plane_buffer(&device, uv_bytes)?;
        let output = frame_output_buffer(&device, output_u32_bytes)?;
        let readback = frame_readback_buffer(&device, output_u32_bytes)?;
        let command_pool = GpuCommandPool::new(device.clone()).map_err(|err| {
            eprintln!("vulkan NV12 frame command pool creation failed: {err:?}");
            FrameProcessorError::Resource(ResourceError::ExportFailed)
        })?;
        let command_buffer = command_pool.allocate_primary().map_err(|err| {
            eprintln!("vulkan NV12 frame command buffer allocation failed: {err:?}");
            FrameProcessorError::Resource(ResourceError::ExportFailed)
        })?;
        let fence = GpuFence::new(device).map_err(|err| {
            eprintln!("vulkan NV12 frame fence creation failed: {err:?}");
            FrameProcessorError::Resource(ResourceError::ExportFailed)
        })?;
        let descriptor = pipeline.write_buffers(&gpu_y, &gpu_uv, &output)?;
        Ok(Self {
            index,
            state: FrameSlotState::Available,
            submitted: None,
            had_submission: false,
            staging_y,
            staging_uv,
            gpu_y,
            gpu_uv,
            output,
            readback,
            command_pool,
            command_buffer,
            fence,
            descriptor,
        })
    }

    fn submit(
        &mut self,
        device: &Arc<GpuDevice>,
        pipeline: &Nv12PipelineState,
        submission: Nv12FrameSubmission<'_>,
        diagnostics: &mut Nv12FrameProcessorDiagnostics,
    ) -> Result<(), FrameProcessorError> {
        if self.state != FrameSlotState::Available {
            return Err(FrameProcessorError::InvalidFrameState);
        }
        self.state = FrameSlotState::Preparing;
        let submit_result = self.submit_inner(device, pipeline, submission, diagnostics);
        if submit_result.is_err() {
            self.state = FrameSlotState::Available;
            self.submitted = None;
        }
        submit_result
    }

    fn submit_inner(
        &mut self,
        device: &Arc<GpuDevice>,
        pipeline: &Nv12PipelineState,
        submission: Nv12FrameSubmission<'_>,
        diagnostics: &mut Nv12FrameProcessorDiagnostics,
    ) -> Result<(), FrameProcessorError> {
        let upload = submission.upload;
        let compact_y_copy_started = Instant::now();
        let y_row_bytes = usize::try_from(upload.y.width_bytes)
            .map_err(|_| FrameProcessorError::Resource(ResourceError::InvalidImageDimensions))?;
        let y_rows = usize::try_from(upload.y.height)
            .map_err(|_| FrameProcessorError::Resource(ResourceError::InvalidImageDimensions))?;
        self.staging_y.write_pitched_rows(
            0,
            upload.y.data,
            upload.y.stride_bytes,
            y_row_bytes,
            y_rows,
            submission.layout.y_stride_bytes,
        )?;
        diagnostics.compact_y_copy_ns = diagnostics
            .compact_y_copy_ns
            .saturating_add(compact_y_copy_started.elapsed().as_nanos());
        let compact_uv_copy_started = Instant::now();
        let uv_row_bytes = usize::try_from(upload.uv.width_bytes)
            .map_err(|_| FrameProcessorError::Resource(ResourceError::InvalidImageDimensions))?;
        let uv_rows = usize::try_from(upload.uv.height)
            .map_err(|_| FrameProcessorError::Resource(ResourceError::InvalidImageDimensions))?;
        self.staging_uv.write_pitched_rows(
            0,
            upload.uv.data,
            upload.uv.stride_bytes,
            uv_row_bytes,
            uv_rows,
            submission.layout.uv_stride_bytes,
        )?;
        diagnostics.compact_uv_copy_ns = diagnostics
            .compact_uv_copy_ns
            .saturating_add(compact_uv_copy_started.elapsed().as_nanos());
        let y_bytes = u64::try_from(submission.layout.y_bytes)
            .map_err(|_| FrameProcessorError::Resource(ResourceError::InvalidBufferSize))?;
        let uv_bytes = u64::try_from(submission.layout.uv_bytes)
            .map_err(|_| FrameProcessorError::Resource(ResourceError::InvalidBufferSize))?;
        let output_copy_bytes = u64::try_from(submission.layout.output_bytes)
            .map_err(|_| FrameProcessorError::Resource(ResourceError::InvalidBufferSize))?;
        let reset_pool_started = Instant::now();
        self.command_pool.reset().map_err(|err| {
            eprintln!("vulkan NV12 frame command pool reset failed: {err:?}");
            FrameProcessorError::Resource(ResourceError::ExportFailed)
        })?;
        diagnostics.command_pool_reset_ns = diagnostics
            .command_pool_reset_ns
            .saturating_add(reset_pool_started.elapsed().as_nanos());
        let reset_fence_started = Instant::now();
        self.fence.reset().map_err(|err| {
            eprintln!("vulkan NV12 frame fence reset failed: {err:?}");
            FrameProcessorError::Resource(ResourceError::ExportFailed)
        })?;
        diagnostics.fence_reset_ns = diagnostics
            .fence_reset_ns
            .saturating_add(reset_fence_started.elapsed().as_nanos());
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let begin_started = Instant::now();
        // SAFETY: command buffer belongs to a reset command pool and is not in
        // use because this slot is Available.
        unsafe {
            device
                .device
                .begin_command_buffer(self.command_buffer, &begin)
        }
        .map_err(|err| {
            eprintln!("vulkan NV12 command buffer begin failed: {err:?}");
            FrameProcessorError::Resource(ResourceError::ExportFailed)
        })?;
        diagnostics.command_begin_ns = diagnostics
            .command_begin_ns
            .saturating_add(begin_started.elapsed().as_nanos());
        let record_started = Instant::now();
        record_nv12_processing_commands(
            device,
            self.command_buffer,
            &self.staging_y,
            &self.staging_uv,
            &self.gpu_y,
            &self.gpu_uv,
            &self.output,
            &self.readback,
            pipeline,
            self.descriptor,
            upload.visible_width,
            upload.visible_height,
            u32::try_from(submission.layout.y_stride_bytes)
                .map_err(|_| FrameProcessorError::Resource(ResourceError::InvalidBufferSize))?,
            u32::try_from(submission.layout.uv_stride_bytes)
                .map_err(|_| FrameProcessorError::Resource(ResourceError::InvalidBufferSize))?,
            y_bytes,
            uv_bytes,
            output_copy_bytes,
            submission.validation_readback,
        );
        diagnostics.command_record_ns = diagnostics
            .command_record_ns
            .saturating_add(record_started.elapsed().as_nanos());
        let end_started = Instant::now();
        // SAFETY: command buffer is currently recording.
        unsafe { device.device.end_command_buffer(self.command_buffer) }.map_err(|err| {
            eprintln!("vulkan NV12 command buffer end failed: {err:?}");
            FrameProcessorError::Resource(ResourceError::ExportFailed)
        })?;
        diagnostics.command_end_ns = diagnostics
            .command_end_ns
            .saturating_add(end_started.elapsed().as_nanos());
        let command_buffers = [self.command_buffer];
        let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
        let submit_started = Instant::now();
        // SAFETY: queue belongs to device, command buffer is executable, and the
        // per-slot fence/resources remain live until completion.
        unsafe {
            device
                .device
                .queue_submit(device.queue, &[submit], self.fence.fence)
        }
        .map_err(|err| {
            eprintln!("vulkan NV12 queue submit failed: {err:?}");
            FrameProcessorError::Resource(ResourceError::ExportFailed)
        })?;
        diagnostics.queue_submit_ns = diagnostics
            .queue_submit_ns
            .saturating_add(submit_started.elapsed().as_nanos());
        self.submitted = Some(Nv12SubmittedFrameInfo {
            token: submission.token,
            identity: submission.identity,
            sequence: submission.sequence,
            width: upload.visible_width,
            height: upload.visible_height,
            layout: submission.layout,
        });
        self.state = FrameSlotState::Submitted;
        self.had_submission = true;
        Ok(())
    }

    fn read_completed_output(&self) -> Result<ProcessedFrameOutput, FrameProcessorError> {
        if self.state != FrameSlotState::Completed {
            return Err(FrameProcessorError::FrameNotComplete);
        }
        let info = self
            .submitted
            .ok_or(FrameProcessorError::UnknownFrameToken)?;
        let output_u16_bytes =
            checked_byte_len(info.layout.output_samples, std::mem::size_of::<u16>())?;
        let bytes = self.readback.read_bytes(0, output_u16_bytes as usize)?;
        let rgba_u16 = read_rgba_u16_from_le_bytes(&bytes, info.layout.output_samples)?;
        if output_u16_bytes != info.layout.output_bytes as u64 {
            return Err(FrameProcessorError::Resource(ResourceError::ExportFailed));
        }
        let checksum = yuv422p10_rgba_u16_checksum(&rgba_u16);
        Ok(ProcessedFrameOutput {
            token: info.token,
            presentation_position: info.identity.presentation_position,
            slot_index: self.index,
            submission_sequence: info.sequence,
            width: info.width,
            height: info.height,
            rgba_u16,
            checksum,
            cpu_surface_bytes: info.layout.cpu_surface_bytes,
            staging_bytes: info.layout.staging_bytes,
            gpu_plane_bytes: info.layout.gpu_plane_bytes,
            output_bytes: info.layout.output_bytes,
        })
    }
}

impl FrameSlot {
    fn new(
        index: usize,
        device: Arc<GpuDevice>,
        pipeline: &Yuv422P10PipelineState,
        layout: Yuv422P10Layout,
    ) -> Result<Self, FrameProcessorError> {
        let y_bytes = checked_byte_len(layout.y_samples, std::mem::size_of::<u32>())?;
        let chroma_bytes = checked_byte_len(layout.chroma_samples, std::mem::size_of::<u32>())?;
        let output_u32_bytes = checked_byte_len(layout.output_samples, std::mem::size_of::<u32>())?;
        let staging_y = frame_staging_buffer(&device, y_bytes)?;
        let staging_cb = frame_staging_buffer(&device, chroma_bytes)?;
        let staging_cr = frame_staging_buffer(&device, chroma_bytes)?;
        let gpu_y = frame_gpu_plane_buffer(&device, y_bytes)?;
        let gpu_cb = frame_gpu_plane_buffer(&device, chroma_bytes)?;
        let gpu_cr = frame_gpu_plane_buffer(&device, chroma_bytes)?;
        let output = frame_output_buffer(&device, output_u32_bytes)?;
        let readback = frame_readback_buffer(&device, output_u32_bytes)?;
        let command_pool = GpuCommandPool::new(device.clone()).map_err(|err| {
            eprintln!("vulkan frame command pool creation failed: {err:?}");
            FrameProcessorError::Resource(ResourceError::ExportFailed)
        })?;
        let command_buffer = command_pool.allocate_primary().map_err(|err| {
            eprintln!("vulkan frame command buffer allocation failed: {err:?}");
            FrameProcessorError::Resource(ResourceError::ExportFailed)
        })?;
        let fence = GpuFence::new(device).map_err(|err| {
            eprintln!("vulkan frame fence creation failed: {err:?}");
            FrameProcessorError::Resource(ResourceError::ExportFailed)
        })?;
        let descriptor = pipeline.write_buffers(&gpu_y, &gpu_cb, &gpu_cr, &output)?;
        Ok(Self {
            index,
            state: FrameSlotState::Available,
            submitted: None,
            had_submission: false,
            staging_y,
            staging_cb,
            staging_cr,
            gpu_y,
            gpu_cb,
            gpu_cr,
            output,
            readback,
            command_pool,
            command_buffer,
            fence,
            descriptor,
        })
    }

    fn submit(
        &mut self,
        device: &Arc<GpuDevice>,
        pipeline: &Yuv422P10PipelineState,
        submission: FrameSubmission<'_>,
    ) -> Result<(), FrameProcessorError> {
        if self.state != FrameSlotState::Available {
            return Err(FrameProcessorError::InvalidFrameState);
        }
        self.state = FrameSlotState::Preparing;
        let submit_result = self.submit_inner(device, pipeline, submission);
        if submit_result.is_err() {
            self.state = FrameSlotState::Available;
            self.submitted = None;
        }
        submit_result
    }

    fn submit_inner(
        &mut self,
        device: &Arc<GpuDevice>,
        pipeline: &Yuv422P10PipelineState,
        submission: FrameSubmission<'_>,
    ) -> Result<(), FrameProcessorError> {
        let upload = submission.upload;
        let y_values = expand_yuv422p10_plane(upload.y)?;
        let cb_values = expand_yuv422p10_plane(upload.cb)?;
        let cr_values = expand_yuv422p10_plane(upload.cr)?;
        let y_bytes = checked_byte_len(y_values.len(), std::mem::size_of::<u32>())?;
        let chroma_bytes = checked_byte_len(cb_values.len(), std::mem::size_of::<u32>())?;
        let output_u32_bytes =
            checked_byte_len(submission.layout.output_samples, std::mem::size_of::<u32>())?;
        self.staging_y.write_bytes(0, cast_u32_slice(&y_values))?;
        self.staging_cb.write_bytes(0, cast_u32_slice(&cb_values))?;
        self.staging_cr.write_bytes(0, cast_u32_slice(&cr_values))?;
        self.command_pool.reset().map_err(|err| {
            eprintln!("vulkan frame command pool reset failed: {err:?}");
            FrameProcessorError::Resource(ResourceError::ExportFailed)
        })?;
        self.fence.reset().map_err(|err| {
            eprintln!("vulkan frame fence reset failed: {err:?}");
            FrameProcessorError::Resource(ResourceError::ExportFailed)
        })?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: command buffer belongs to a reset command pool and is not in
        // use because this slot is Available.
        unsafe {
            device
                .device
                .begin_command_buffer(self.command_buffer, &begin)
        }
        .map_err(|err| {
            eprintln!("vulkan frame command buffer begin failed: {err:?}");
            FrameProcessorError::Resource(ResourceError::ExportFailed)
        })?;
        record_yuv422p10_processing_commands(
            device,
            self.command_buffer,
            &self.staging_y,
            &self.staging_cb,
            &self.staging_cr,
            &self.gpu_y,
            &self.gpu_cb,
            &self.gpu_cr,
            &self.output,
            &self.readback,
            pipeline,
            self.descriptor,
            upload.width,
            upload.height,
            y_bytes,
            chroma_bytes,
            output_u32_bytes,
        );
        // SAFETY: command buffer is currently recording.
        unsafe { device.device.end_command_buffer(self.command_buffer) }.map_err(|err| {
            eprintln!("vulkan frame command buffer end failed: {err:?}");
            FrameProcessorError::Resource(ResourceError::ExportFailed)
        })?;
        let command_buffers = [self.command_buffer];
        let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
        // SAFETY: queue belongs to device, command buffer is executable, and the
        // per-slot fence/resources remain live until completion.
        unsafe {
            device
                .device
                .queue_submit(device.queue, &[submit], self.fence.fence)
        }
        .map_err(|err| {
            eprintln!("vulkan frame queue submit failed: {err:?}");
            FrameProcessorError::Resource(ResourceError::ExportFailed)
        })?;
        self.submitted = Some(SubmittedFrameInfo {
            token: submission.token,
            identity: submission.identity,
            sequence: submission.sequence,
            width: upload.width,
            height: upload.height,
            layout: submission.layout,
        });
        self.state = FrameSlotState::Submitted;
        self.had_submission = true;
        Ok(())
    }

    fn read_completed_output(&self) -> Result<ProcessedFrameOutput, FrameProcessorError> {
        if self.state != FrameSlotState::Completed {
            return Err(FrameProcessorError::FrameNotComplete);
        }
        let info = self
            .submitted
            .ok_or(FrameProcessorError::UnknownFrameToken)?;
        let output_u32_bytes =
            checked_byte_len(info.layout.output_samples, std::mem::size_of::<u32>())?;
        let output_u16_bytes =
            checked_byte_len(info.layout.output_samples, std::mem::size_of::<u16>())?;
        let bytes = self.readback.read_bytes(0, output_u32_bytes as usize)?;
        let rgba_u16 = read_rgba_u16_from_u32_bytes(&bytes, info.layout.output_samples)?;
        if output_u16_bytes != info.layout.output_bytes as u64 {
            return Err(FrameProcessorError::Resource(ResourceError::ExportFailed));
        }
        let checksum = yuv422p10_rgba_u16_checksum(&rgba_u16);
        Ok(ProcessedFrameOutput {
            token: info.token,
            presentation_position: info.identity.presentation_position,
            slot_index: self.index,
            submission_sequence: info.sequence,
            width: info.width,
            height: info.height,
            rgba_u16,
            checksum,
            cpu_surface_bytes: info.layout.cpu_surface_bytes,
            staging_bytes: info.layout.staging_bytes,
            gpu_plane_bytes: info.layout.gpu_plane_bytes,
            output_bytes: info.layout.output_bytes,
        })
    }
}

fn frame_processor_resource_error(err: FrameProcessorError) -> ResourceError {
    match err {
        FrameProcessorError::Resource(err) => err,
        FrameProcessorError::NoFrameSlotAvailable
        | FrameProcessorError::UnknownFrameToken
        | FrameProcessorError::FrameNotComplete
        | FrameProcessorError::InvalidFrameState => ResourceError::ExportFailed,
    }
}

fn validate_frame_processor_config(
    config: GpuFrameProcessorConfig,
) -> Result<(), FrameProcessorError> {
    if config.slot_count == 0 || config.slot_count > 8 {
        return Err(FrameProcessorError::Resource(
            ResourceError::InvalidBufferSize,
        ));
    }
    yuv422p10_layout_for_dimensions(config.width, config.height, 0)?;
    Ok(())
}

fn validate_nv12_frame_processor_config(
    config: Nv12FrameProcessorConfig,
) -> Result<(), FrameProcessorError> {
    if config.slot_count == 0 || config.slot_count > 8 {
        return Err(FrameProcessorError::Resource(
            ResourceError::InvalidBufferSize,
        ));
    }
    nv12_layout_for_dimensions(
        config.coded_width,
        config.coded_height,
        config.visible_width,
        config.visible_height,
        0,
    )?;
    Ok(())
}

fn yuv422p10_layout_for_dimensions(
    width: u32,
    height: u32,
    cpu_surface_bytes: usize,
) -> Result<Yuv422P10Layout, ResourceError> {
    if width == 0
        || height == 0
        || width > MAX_VIDEO_SURFACE_WIDTH
        || height > MAX_VIDEO_SURFACE_HEIGHT
        || !width.is_multiple_of(2)
    {
        return Err(ResourceError::InvalidImageDimensions);
    }
    let y_samples = checked_sample_count(width, height)?;
    let chroma_samples = checked_sample_count(width / 2, height)?;
    let output_samples = y_samples
        .checked_mul(YUV422P10_OUTPUT_COMPONENTS)
        .ok_or(ResourceError::InvalidImageDimensions)?;
    let staging_samples = y_samples
        .checked_add(chroma_samples)
        .and_then(|total| total.checked_add(chroma_samples))
        .ok_or(ResourceError::InvalidImageDimensions)?;
    let staging_bytes = staging_samples
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(ResourceError::InvalidImageDimensions)?;
    let output_bytes = output_samples
        .checked_mul(YUV422P10_OUTPUT_BYTES_PER_COMPONENT)
        .ok_or(ResourceError::InvalidImageDimensions)?;
    Ok(Yuv422P10Layout {
        y_samples,
        chroma_samples,
        output_samples,
        cpu_surface_bytes,
        staging_bytes,
        gpu_plane_bytes: staging_bytes,
        output_bytes,
    })
}

fn frame_staging_buffer(device: &Arc<GpuDevice>, size: u64) -> Result<GpuBuffer, ResourceError> {
    GpuBuffer::new(
        device.clone(),
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        None,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
        vk::MemoryPropertyFlags::HOST_COHERENT,
        false,
    )
}

fn frame_gpu_plane_buffer(device: &Arc<GpuDevice>, size: u64) -> Result<GpuBuffer, ResourceError> {
    GpuBuffer::new(
        device.clone(),
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
        None,
        vk::MemoryPropertyFlags::empty(),
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        false,
    )
}

fn frame_output_buffer(device: &Arc<GpuDevice>, size: u64) -> Result<GpuBuffer, ResourceError> {
    GpuBuffer::new(
        device.clone(),
        size,
        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
        None,
        vk::MemoryPropertyFlags::empty(),
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        false,
    )
}

fn frame_readback_buffer(device: &Arc<GpuDevice>, size: u64) -> Result<GpuBuffer, ResourceError> {
    GpuBuffer::new(
        device.clone(),
        size,
        vk::BufferUsageFlags::TRANSFER_DST,
        None,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
        vk::MemoryPropertyFlags::HOST_COHERENT,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn record_yuv422p10_processing_commands(
    device: &GpuDevice,
    command_buffer: vk::CommandBuffer,
    staging_y: &GpuBuffer,
    staging_cb: &GpuBuffer,
    staging_cr: &GpuBuffer,
    gpu_y: &GpuBuffer,
    gpu_cb: &GpuBuffer,
    gpu_cr: &GpuBuffer,
    output: &GpuBuffer,
    readback: &GpuBuffer,
    pipeline: &Yuv422P10PipelineState,
    descriptor: DescriptorSet,
    width: u32,
    height: u32,
    y_bytes: u64,
    chroma_bytes: u64,
    output_u32_bytes: u64,
) {
    buffer_barrier(
        device,
        command_buffer,
        &[gpu_y, gpu_cb, gpu_cr],
        vk::AccessFlags::SHADER_READ,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
    );
    buffer_barrier(
        device,
        command_buffer,
        &[output],
        vk::AccessFlags::TRANSFER_READ,
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    );
    buffer_barrier(
        device,
        command_buffer,
        &[readback],
        vk::AccessFlags::HOST_READ,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::HOST,
        vk::PipelineStageFlags::TRANSFER,
    );
    copy_buffer(command_buffer, staging_y, gpu_y, y_bytes);
    copy_buffer(command_buffer, staging_cb, gpu_cb, chroma_bytes);
    copy_buffer(command_buffer, staging_cr, gpu_cr, chroma_bytes);
    buffer_barrier(
        device,
        command_buffer,
        &[gpu_y, gpu_cb, gpu_cr],
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    );
    let params = Yuv422P10ShaderParams {
        width,
        height,
        y_stride_samples: width,
        chroma_stride_samples: width / 2,
    };
    // SAFETY: command buffer is recording; pipeline, descriptor set and buffers
    // are live for the submission. Push constants match the fixed shader's
    // 16-byte parameter block.
    unsafe {
        device.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.pipeline,
        );
        device.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            0,
            &[descriptor.set],
            &[],
        );
        device.device.cmd_push_constants(
            command_buffer,
            pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            params.as_bytes(),
        );
        device.device.cmd_dispatch(
            command_buffer,
            width.div_ceil(YUV422P10_LOCAL_SIZE_X),
            height.div_ceil(YUV422P10_LOCAL_SIZE_Y),
            1,
        );
    }
    buffer_barrier(
        device,
        command_buffer,
        &[output],
        vk::AccessFlags::SHADER_WRITE,
        vk::AccessFlags::TRANSFER_READ,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
    );
    copy_buffer(command_buffer, output, readback, output_u32_bytes);
    buffer_barrier(
        device,
        command_buffer,
        &[readback],
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::HOST_READ,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::HOST,
    );
}

#[allow(clippy::too_many_arguments)]
fn record_nv12_processing_commands(
    device: &GpuDevice,
    command_buffer: vk::CommandBuffer,
    staging_y: &GpuBuffer,
    staging_uv: &GpuBuffer,
    gpu_y: &GpuBuffer,
    gpu_uv: &GpuBuffer,
    output: &GpuBuffer,
    readback: &GpuBuffer,
    pipeline: &Nv12PipelineState,
    descriptor: DescriptorSet,
    visible_width: u32,
    visible_height: u32,
    y_stride_bytes: u32,
    uv_stride_bytes: u32,
    y_bytes: u64,
    uv_bytes: u64,
    output_copy_bytes: u64,
    validation_readback: bool,
) {
    buffer_barrier(
        device,
        command_buffer,
        &[gpu_y, gpu_uv],
        vk::AccessFlags::SHADER_READ,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
    );
    buffer_barrier(
        device,
        command_buffer,
        &[output],
        vk::AccessFlags::TRANSFER_READ,
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    );
    if validation_readback {
        buffer_barrier(
            device,
            command_buffer,
            &[readback],
            vk::AccessFlags::HOST_READ,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
        );
    }
    copy_buffer(command_buffer, staging_y, gpu_y, y_bytes);
    copy_buffer(command_buffer, staging_uv, gpu_uv, uv_bytes);
    buffer_barrier(
        device,
        command_buffer,
        &[gpu_y, gpu_uv],
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    );
    let params = Nv12ShaderParams {
        width: visible_width,
        height: visible_height,
        y_stride_bytes,
        uv_stride_bytes,
    };
    // SAFETY: command buffer is recording; pipeline, descriptor set and buffers
    // are live for the submission. Push constants match the fixed shader's
    // 16-byte parameter block.
    unsafe {
        device.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.pipeline,
        );
        device.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            0,
            &[descriptor.set],
            &[],
        );
        device.device.cmd_push_constants(
            command_buffer,
            pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            params.as_bytes(),
        );
        device.device.cmd_dispatch(
            command_buffer,
            visible_width.div_ceil(YUV422P10_LOCAL_SIZE_X),
            visible_height.div_ceil(YUV422P10_LOCAL_SIZE_Y),
            1,
        );
    }
    if validation_readback {
        buffer_barrier(
            device,
            command_buffer,
            &[output],
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
        );
        copy_buffer(command_buffer, output, readback, output_copy_bytes);
        buffer_barrier(
            device,
            command_buffer,
            &[readback],
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::HOST_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
        );
    }
}

fn read_rgba_u16_from_u32_bytes(
    bytes: &[u8],
    expected_samples: usize,
) -> Result<Vec<u16>, FrameProcessorError> {
    let mut rgba_u16 = Vec::with_capacity(expected_samples);
    let (chunks, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(FrameProcessorError::Resource(ResourceError::ExportFailed));
    }
    for chunk in chunks {
        let value = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        rgba_u16.push(
            u16::try_from(value)
                .map_err(|_| FrameProcessorError::Resource(ResourceError::ExportFailed))?,
        );
    }
    if rgba_u16.len() != expected_samples {
        return Err(FrameProcessorError::Resource(ResourceError::ExportFailed));
    }
    Ok(rgba_u16)
}

fn read_rgba_u16_from_le_bytes(
    bytes: &[u8],
    expected_samples: usize,
) -> Result<Vec<u16>, FrameProcessorError> {
    let expected_bytes = expected_samples
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or(FrameProcessorError::Resource(ResourceError::ExportFailed))?;
    if bytes.len() != expected_bytes {
        return Err(FrameProcessorError::Resource(ResourceError::ExportFailed));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn checked_byte_len(count: usize, bytes_per_item: usize) -> Result<u64, ResourceError> {
    let bytes = count
        .checked_mul(bytes_per_item)
        .ok_or(ResourceError::InvalidBufferSize)?;
    u64::try_from(bytes).map_err(|_| ResourceError::InvalidBufferSize)
}

fn copy_buffer(command_buffer: vk::CommandBuffer, src: &GpuBuffer, dst: &GpuBuffer, size: u64) {
    let copy = vk::BufferCopy::default().size(size);
    // SAFETY: command buffer is recording, buffers are live, and caller uses
    // QGS-created buffers with the required transfer usages.
    unsafe {
        src.device
            .device
            .cmd_copy_buffer(command_buffer, src.buffer, dst.buffer, &[copy]);
    }
}

fn buffer_barrier(
    device: &GpuDevice,
    command_buffer: vk::CommandBuffer,
    buffers: &[&GpuBuffer],
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
) {
    let barriers = buffers
        .iter()
        .map(|buffer| {
            vk::BufferMemoryBarrier::default()
                .src_access_mask(src_access)
                .dst_access_mask(dst_access)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(buffer.buffer)
                .offset(0)
                .size(buffer.size)
        })
        .collect::<Vec<_>>();
    // SAFETY: command buffer is recording. Barriers reference live buffers
    // owned by the same device and cover their full allocation-backed ranges.
    unsafe {
        device.device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &barriers,
            &[],
        );
    }
}

struct ImageTransition {
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
}

fn transition_image(
    device: &GpuDevice,
    command_buffer: vk::CommandBuffer,
    transition: ImageTransition,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(transition.src_access)
        .dst_access_mask(transition.dst_access)
        .old_layout(transition.old_layout)
        .new_layout(transition.new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(transition.image)
        .subresource_range(color_subresource_range());
    // SAFETY: command buffer is recording; barrier references a live image and
    // a single color subresource range used by QGS RGBA images.
    unsafe {
        device.device.cmd_pipeline_barrier(
            command_buffer,
            transition.src_stage,
            transition.dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}

#[derive(Debug)]
struct ComputePipelineState {
    device: Arc<GpuDevice>,
    shader: vk::ShaderModule,
    descriptor_set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_type: vk::DescriptorType,
}

impl ComputePipelineState {
    fn new_storage_buffer(device: Arc<GpuDevice>) -> Result<Self, ResourceError> {
        Self::new(
            device,
            &COMPUTE_INCREMENT_SHADER,
            vk::DescriptorType::STORAGE_BUFFER,
        )
    }

    fn new_storage_image(device: Arc<GpuDevice>) -> Result<Self, ResourceError> {
        Self::new(
            device,
            &IMAGE_INVERT_SHADER,
            vk::DescriptorType::STORAGE_IMAGE,
        )
    }

    fn new(
        device: Arc<GpuDevice>,
        shader_code: &[u32],
        descriptor_type: vk::DescriptorType,
    ) -> Result<Self, ResourceError> {
        let shader_info = vk::ShaderModuleCreateInfo::default().code(shader_code);
        // SAFETY: shader_code contains embedded SPIR-V words controlled by QGS.
        let shader =
            unsafe { device.device.create_shader_module(&shader_info, None) }.map_err(|err| {
                eprintln!("vulkan shader module creation failed: {err:?}");
                ResourceError::ExportFailed
            })?;
        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(descriptor_type)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let descriptor_set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding);
        // SAFETY: descriptor set layout info references local binding array for call.
        let descriptor_set_layout = unsafe {
            device
                .device
                .create_descriptor_set_layout(&descriptor_set_layout_info, None)
        }
        .map_err(|err| {
            eprintln!("vulkan descriptor set layout creation failed: {err:?}");
            // SAFETY: shader was created above and is owned here on failure.
            unsafe {
                device.device.destroy_shader_module(shader, None);
            }
            ResourceError::ExportFailed
        })?;
        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        // SAFETY: pipeline layout info references live descriptor set layout.
        let layout = unsafe {
            device
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .map_err(|err| {
            eprintln!("vulkan pipeline layout creation failed: {err:?}");
            // SAFETY: objects are live and owned by this constructor.
            unsafe {
                device
                    .device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
                device.device.destroy_shader_module(shader, None);
            }
            ResourceError::ExportFailed
        })?;
        let entry_point = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(entry_point);
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        // SAFETY: shader module and pipeline layout are live and compatible with
        // the fixed QGS descriptor contract.
        let pipeline = unsafe {
            device.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        }
        .map_err(|(_, err)| {
            eprintln!("vulkan compute pipeline creation failed: {err:?}");
            // SAFETY: objects are live and owned by this constructor.
            unsafe {
                device.device.destroy_pipeline_layout(layout, None);
                device
                    .device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
                device.device.destroy_shader_module(shader, None);
            }
            ResourceError::ExportFailed
        })?[0];
        let pool_size = [vk::DescriptorPoolSize::default()
            .ty(descriptor_type)
            .descriptor_count(1)];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_size);
        // SAFETY: descriptor pool info is valid for one fixed descriptor set.
        let descriptor_pool = unsafe {
            device
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)
        }
        .map_err(|err| {
            eprintln!("vulkan descriptor pool creation failed: {err:?}");
            // SAFETY: objects are live and owned by this constructor.
            unsafe {
                device.device.destroy_pipeline(pipeline, None);
                device.device.destroy_pipeline_layout(layout, None);
                device
                    .device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
                device.device.destroy_shader_module(shader, None);
            }
            ResourceError::ExportFailed
        })?;

        Ok(Self {
            device,
            shader,
            descriptor_set_layout,
            layout,
            pipeline,
            descriptor_pool,
            descriptor_type,
        })
    }

    fn allocate_set(&self) -> Result<vk::DescriptorSet, ResourceError> {
        let set_layouts = [self.descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&set_layouts);
        // SAFETY: descriptor pool and set layout are live.
        let sets = unsafe { self.device.device.allocate_descriptor_sets(&allocate_info) }.map_err(
            |err| {
                eprintln!("vulkan descriptor set allocation failed: {err:?}");
                ResourceError::ExportFailed
            },
        )?;
        Ok(sets[0])
    }

    fn write_storage_buffer(&self, buffer: &GpuBuffer) -> Result<DescriptorSet, ResourceError> {
        if self.descriptor_type != vk::DescriptorType::STORAGE_BUFFER {
            return Err(ResourceError::ExportFailed);
        }
        let set = self.allocate_set()?;
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer.buffer)
            .offset(0)
            .range(buffer.size)];
        let write = [vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_info)];
        // SAFETY: descriptor set is allocated from this pool, and buffer info
        // points to a live buffer for the call.
        unsafe {
            self.device.device.update_descriptor_sets(&write, &[]);
        }
        Ok(DescriptorSet { set })
    }

    fn write_storage_image(&self, view: &GpuImageView) -> Result<DescriptorSet, ResourceError> {
        if self.descriptor_type != vk::DescriptorType::STORAGE_IMAGE {
            return Err(ResourceError::ExportFailed);
        }
        let set = self.allocate_set()?;
        let image_info = [vk::DescriptorImageInfo::default()
            .image_view(view.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let write = [vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_info)];
        // SAFETY: descriptor set is allocated from this pool, and image view is
        // live for the duration of the call and subsequent submission.
        unsafe {
            self.device.device.update_descriptor_sets(&write, &[]);
        }
        Ok(DescriptorSet { set })
    }
}

impl Drop for ComputePipelineState {
    fn drop(&mut self) {
        // SAFETY: ComputePipelineState owns these objects. They are destroyed in
        // dependency order after submissions using them have completed.
        unsafe {
            self.device
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.device.destroy_pipeline(self.pipeline, None);
            self.device
                .device
                .destroy_pipeline_layout(self.layout, None);
            self.device
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.device.destroy_shader_module(self.shader, None);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Yuv422P10ShaderParams {
    width: u32,
    height: u32,
    y_stride_samples: u32,
    chroma_stride_samples: u32,
}

impl Yuv422P10ShaderParams {
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: Yuv422P10ShaderParams is repr(C), contains only u32 fields,
        // and has no padding-dependent validity. The returned slice is tied to
        // self and exactly covers the push-constant struct.
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self).cast::<u8>(),
                std::mem::size_of::<Self>(),
            )
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Nv12ShaderParams {
    width: u32,
    height: u32,
    y_stride_bytes: u32,
    uv_stride_bytes: u32,
}

impl Nv12ShaderParams {
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: Nv12ShaderParams is repr(C), contains only u32 fields, and
        // has no padding-dependent validity. The returned slice is tied to self
        // and exactly covers the push-constant struct.
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self).cast::<u8>(),
                std::mem::size_of::<Self>(),
            )
        }
    }
}

#[derive(Debug)]
struct Yuv422P10PipelineState {
    device: Arc<GpuDevice>,
    shader: vk::ShaderModule,
    descriptor_set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
}

impl Yuv422P10PipelineState {
    fn new(device: Arc<GpuDevice>, max_sets: u32) -> Result<Self, ResourceError> {
        if max_sets == 0 {
            return Err(ResourceError::InvalidBufferSize);
        }
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&YUV422P10_TO_RGBA_U16_SHADER);
        // SAFETY: shader code is fixed QGS-controlled SPIR-V compiled for the
        // descriptor/push-constant contract below.
        let shader =
            unsafe { device.device.create_shader_module(&shader_info, None) }.map_err(|err| {
                eprintln!("vulkan YUV shader module creation failed: {err:?}");
                ResourceError::ExportFailed
            })?;
        let bindings = [
            storage_buffer_binding(0),
            storage_buffer_binding(1),
            storage_buffer_binding(2),
            storage_buffer_binding(3),
        ];
        let descriptor_set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: descriptor set layout info references the local bindings for
        // the duration of this call.
        let descriptor_set_layout = unsafe {
            device
                .device
                .create_descriptor_set_layout(&descriptor_set_layout_info, None)
        }
        .map_err(|err| {
            eprintln!("vulkan YUV descriptor set layout creation failed: {err:?}");
            // SAFETY: shader was created above and is still owned here.
            unsafe {
                device.device.destroy_shader_module(shader, None);
            }
            ResourceError::ExportFailed
        })?;
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<Yuv422P10ShaderParams>() as u32)];
        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        // SAFETY: pipeline layout info references live descriptor set layout and
        // fixed push-constant range.
        let layout = unsafe {
            device
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .map_err(|err| {
            eprintln!("vulkan YUV pipeline layout creation failed: {err:?}");
            // SAFETY: objects are live and owned by this constructor.
            unsafe {
                device
                    .device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
                device.device.destroy_shader_module(shader, None);
            }
            ResourceError::ExportFailed
        })?;
        let entry_point = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(entry_point);
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        // SAFETY: shader module and pipeline layout are live and match the
        // fixed QGS YUV422P10 descriptor contract.
        let pipeline = unsafe {
            device.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        }
        .map_err(|(_, err)| {
            eprintln!("vulkan YUV compute pipeline creation failed: {err:?}");
            // SAFETY: objects are live and owned by this constructor.
            unsafe {
                device.device.destroy_pipeline_layout(layout, None);
                device
                    .device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
                device.device.destroy_shader_module(shader, None);
            }
            ResourceError::ExportFailed
        })?[0];
        let pool_size = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(
                max_sets
                    .checked_mul(4)
                    .ok_or(ResourceError::InvalidBufferSize)?,
            )];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(&pool_size);
        // SAFETY: descriptor pool info is valid for max_sets fixed four-binding sets.
        let descriptor_pool = unsafe {
            device
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)
        }
        .map_err(|err| {
            eprintln!("vulkan YUV descriptor pool creation failed: {err:?}");
            // SAFETY: objects are live and owned by this constructor.
            unsafe {
                device.device.destroy_pipeline(pipeline, None);
                device.device.destroy_pipeline_layout(layout, None);
                device
                    .device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
                device.device.destroy_shader_module(shader, None);
            }
            ResourceError::ExportFailed
        })?;

        Ok(Self {
            device,
            shader,
            descriptor_set_layout,
            layout,
            pipeline,
            descriptor_pool,
        })
    }

    fn write_buffers(
        &self,
        y: &GpuBuffer,
        cb: &GpuBuffer,
        cr: &GpuBuffer,
        output: &GpuBuffer,
    ) -> Result<DescriptorSet, ResourceError> {
        let set_layouts = [self.descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&set_layouts);
        // SAFETY: descriptor pool and set layout are live.
        let sets = unsafe { self.device.device.allocate_descriptor_sets(&allocate_info) }.map_err(
            |err| {
                eprintln!("vulkan YUV descriptor set allocation failed: {err:?}");
                ResourceError::ExportFailed
            },
        )?;
        let set = sets[0];
        let buffer_infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(y.buffer)
                .offset(0)
                .range(y.size),
            vk::DescriptorBufferInfo::default()
                .buffer(cb.buffer)
                .offset(0)
                .range(cb.size),
            vk::DescriptorBufferInfo::default()
                .buffer(cr.buffer)
                .offset(0)
                .range(cr.size),
            vk::DescriptorBufferInfo::default()
                .buffer(output.buffer)
                .offset(0)
                .range(output.size),
        ];
        let writes = [
            storage_buffer_write(set, 0, &buffer_infos[0..1]),
            storage_buffer_write(set, 1, &buffer_infos[1..2]),
            storage_buffer_write(set, 2, &buffer_infos[2..3]),
            storage_buffer_write(set, 3, &buffer_infos[3..4]),
        ];
        // SAFETY: descriptor set is allocated from this pool, and buffer infos
        // reference live buffers for the call and subsequent submission.
        unsafe {
            self.device.device.update_descriptor_sets(&writes, &[]);
        }
        Ok(DescriptorSet { set })
    }
}

impl Drop for Yuv422P10PipelineState {
    fn drop(&mut self) {
        // SAFETY: Yuv422P10PipelineState owns these objects and they are
        // destroyed in dependency order after submissions complete.
        unsafe {
            self.device
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.device.destroy_pipeline(self.pipeline, None);
            self.device
                .device
                .destroy_pipeline_layout(self.layout, None);
            self.device
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.device.destroy_shader_module(self.shader, None);
        }
    }
}

#[derive(Debug)]
struct Nv12PipelineState {
    device: Arc<GpuDevice>,
    shader: vk::ShaderModule,
    descriptor_set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
}

impl Nv12PipelineState {
    fn new(device: Arc<GpuDevice>, max_sets: u32) -> Result<Self, ResourceError> {
        if max_sets == 0 {
            return Err(ResourceError::InvalidBufferSize);
        }
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&NV12_TO_RGBA_U16_SHADER);
        // SAFETY: shader code is fixed QGS-controlled SPIR-V compiled for the
        // descriptor/push-constant contract below.
        let shader =
            unsafe { device.device.create_shader_module(&shader_info, None) }.map_err(|err| {
                eprintln!("vulkan NV12 shader module creation failed: {err:?}");
                ResourceError::ExportFailed
            })?;
        let bindings = [
            storage_buffer_binding(0),
            storage_buffer_binding(1),
            storage_buffer_binding(2),
        ];
        let descriptor_set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: descriptor set layout info references the local bindings for
        // the duration of this call.
        let descriptor_set_layout = unsafe {
            device
                .device
                .create_descriptor_set_layout(&descriptor_set_layout_info, None)
        }
        .map_err(|err| {
            eprintln!("vulkan NV12 descriptor set layout creation failed: {err:?}");
            // SAFETY: shader was created above and is still owned here.
            unsafe {
                device.device.destroy_shader_module(shader, None);
            }
            ResourceError::ExportFailed
        })?;
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<Nv12ShaderParams>() as u32)];
        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        // SAFETY: pipeline layout info references live descriptor set layout and
        // fixed push-constant range.
        let layout = unsafe {
            device
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .map_err(|err| {
            eprintln!("vulkan NV12 pipeline layout creation failed: {err:?}");
            // SAFETY: objects are live and owned by this constructor.
            unsafe {
                device
                    .device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
                device.device.destroy_shader_module(shader, None);
            }
            ResourceError::ExportFailed
        })?;
        let entry_point = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(entry_point);
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        // SAFETY: shader module and pipeline layout are live and match the
        // fixed QGS NV12 descriptor contract.
        let pipeline = unsafe {
            device.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        }
        .map_err(|(_, err)| {
            eprintln!("vulkan NV12 compute pipeline creation failed: {err:?}");
            // SAFETY: objects are live and owned by this constructor.
            unsafe {
                device.device.destroy_pipeline_layout(layout, None);
                device
                    .device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
                device.device.destroy_shader_module(shader, None);
            }
            ResourceError::ExportFailed
        })?[0];
        let pool_size = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(
                max_sets
                    .checked_mul(3)
                    .ok_or(ResourceError::InvalidBufferSize)?,
            )];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(&pool_size);
        // SAFETY: descriptor pool info is valid for max_sets fixed three-binding sets.
        let descriptor_pool = unsafe {
            device
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)
        }
        .map_err(|err| {
            eprintln!("vulkan NV12 descriptor pool creation failed: {err:?}");
            // SAFETY: objects are live and owned by this constructor.
            unsafe {
                device.device.destroy_pipeline(pipeline, None);
                device.device.destroy_pipeline_layout(layout, None);
                device
                    .device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
                device.device.destroy_shader_module(shader, None);
            }
            ResourceError::ExportFailed
        })?;

        Ok(Self {
            device,
            shader,
            descriptor_set_layout,
            layout,
            pipeline,
            descriptor_pool,
        })
    }

    fn write_buffers(
        &self,
        y: &GpuBuffer,
        uv: &GpuBuffer,
        output: &GpuBuffer,
    ) -> Result<DescriptorSet, ResourceError> {
        let set_layouts = [self.descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&set_layouts);
        // SAFETY: descriptor pool and set layout are live.
        let sets = unsafe { self.device.device.allocate_descriptor_sets(&allocate_info) }.map_err(
            |err| {
                eprintln!("vulkan NV12 descriptor set allocation failed: {err:?}");
                ResourceError::ExportFailed
            },
        )?;
        let set = sets[0];
        let buffer_infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(y.buffer)
                .offset(0)
                .range(y.size),
            vk::DescriptorBufferInfo::default()
                .buffer(uv.buffer)
                .offset(0)
                .range(uv.size),
            vk::DescriptorBufferInfo::default()
                .buffer(output.buffer)
                .offset(0)
                .range(output.size),
        ];
        let writes = [
            storage_buffer_write(set, 0, &buffer_infos[0..1]),
            storage_buffer_write(set, 1, &buffer_infos[1..2]),
            storage_buffer_write(set, 2, &buffer_infos[2..3]),
        ];
        // SAFETY: descriptor set is allocated from this pool, and buffer infos
        // reference live buffers for the call and subsequent submission.
        unsafe {
            self.device.device.update_descriptor_sets(&writes, &[]);
        }
        Ok(DescriptorSet { set })
    }
}

impl Drop for Nv12PipelineState {
    fn drop(&mut self) {
        // SAFETY: Nv12PipelineState owns these objects and they are destroyed in
        // dependency order after submissions complete.
        unsafe {
            self.device
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.device.destroy_pipeline(self.pipeline, None);
            self.device
                .device
                .destroy_pipeline_layout(self.layout, None);
            self.device
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.device.destroy_shader_module(self.shader, None);
        }
    }
}

fn storage_buffer_binding(binding: u32) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

fn storage_buffer_write<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    info: &'a [vk::DescriptorBufferInfo],
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(info)
}

#[derive(Clone, Copy, Debug)]
struct DescriptorSet {
    set: vk::DescriptorSet,
}

fn cast_u32_slice(values: &[u32]) -> &[u8] {
    // SAFETY: u32 has no invalid byte patterns, and the returned byte slice is
    // tied to values' lifetime with length exactly values.len() * size_of::<u32>().
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn device_matches_export_source(left: &DeviceDesc, right: &DeviceDesc) -> bool {
    left.backend == right.backend
        && left.vendor_id == right.vendor_id
        && left.device_id == right.device_id
        && left.name == right.name
}

fn choose_memory_type_from_properties(
    properties: &vk::PhysicalDeviceMemoryProperties,
    memory_type_bits: u32,
    required: vk::MemoryPropertyFlags,
    preferred: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let mut fallback = None;
    for index in 0..properties.memory_type_count {
        if (memory_type_bits & (1_u32 << index)) == 0 {
            continue;
        }
        let flags = properties.memory_types[index as usize].property_flags;
        if !flags.contains(required) {
            continue;
        }
        if flags.contains(preferred) {
            return Some(index);
        }
        fallback.get_or_insert(index);
    }
    fallback
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory_properties(types: &[vk::MemoryPropertyFlags]) -> vk::PhysicalDeviceMemoryProperties {
        let mut properties = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: types.len() as u32,
            ..Default::default()
        };
        for (index, flags) in types.iter().enumerate() {
            properties.memory_types[index].property_flags = *flags;
        }
        properties
    }

    #[test]
    fn memory_type_selection_prefers_preferred_flags() {
        let properties = memory_properties(&[
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ]);

        assert_eq!(
            choose_memory_type_from_properties(
                &properties,
                0b111,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
                vk::MemoryPropertyFlags::HOST_COHERENT,
            ),
            Some(1)
        );
    }

    #[test]
    fn memory_type_selection_falls_back_to_required_only() {
        let properties = memory_properties(&[
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ]);

        assert_eq!(
            choose_memory_type_from_properties(
                &properties,
                0b11,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
                vk::MemoryPropertyFlags::HOST_COHERENT,
            ),
            Some(0)
        );
    }

    #[test]
    fn memory_type_selection_respects_type_bits() {
        let properties = memory_properties(&[
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ]);

        assert_eq!(
            choose_memory_type_from_properties(
                &properties,
                0b10,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
                vk::MemoryPropertyFlags::HOST_COHERENT,
            ),
            None
        );
    }

    #[test]
    fn qgs_api_version_maps_vulkan_components() {
        let version = to_qgs_api_version(vk::make_api_version(0, 1, 3, 281));
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 3);
        assert_eq!(version.patch, 281);
    }

    fn tiny_yuv422p10_upload<'a>(
        y: &'a [u8],
        cb: &'a [u8],
        cr: &'a [u8],
        stride_y: usize,
        stride_c: usize,
    ) -> Yuv422P10Upload<'a> {
        Yuv422P10Upload {
            device_id: DeviceId::new(1).expect("device"),
            width: 2,
            height: 1,
            y: Yuv422P10Plane {
                width_samples: 2,
                height: 1,
                stride_bytes: stride_y,
                data: y,
            },
            cb: Yuv422P10Plane {
                width_samples: 1,
                height: 1,
                stride_bytes: stride_c,
                data: cb,
            },
            cr: Yuv422P10Plane {
                width_samples: 1,
                height: 1,
                stride_bytes: stride_c,
                data: cr,
            },
            conversion: YcbcrConversion::Rec709Limited,
        }
    }

    #[test]
    fn validates_yuv422p10_plane_dimensions_and_padding() {
        let y = [64_u16.to_le_bytes(), 940_u16.to_le_bytes(), [0xaa, 0xbb]].concat();
        let cb = [512_u16.to_le_bytes(), [0xcc, 0xdd]].concat();
        let cr = [512_u16.to_le_bytes(), [0xee, 0xff]].concat();
        let upload = tiny_yuv422p10_upload(&y, &cb, &cr, 6, 4);

        let layout = validate_yuv422p10_upload(&upload).expect("valid upload");

        assert_eq!(layout.y_samples, 2);
        assert_eq!(layout.chroma_samples, 1);
        assert_eq!(layout.cpu_surface_bytes, 14);
        assert_eq!(
            expand_yuv422p10_plane(upload.y).expect("expand"),
            vec![64, 940]
        );
    }

    #[test]
    fn rejects_invalid_yuv422p10_uploads() {
        let y = [0_u8; 4];
        let c = [0_u8; 2];
        let mut upload = tiny_yuv422p10_upload(&y, &c, &c, 4, 2);

        upload.width = 3;
        assert!(matches!(
            validate_yuv422p10_upload(&upload),
            Err(ResourceError::InvalidImageDimensions)
        ));

        upload.width = 2;
        upload.cb.stride_bytes = 1;
        assert!(matches!(
            validate_yuv422p10_upload(&upload),
            Err(ResourceError::InvalidImageDimensions)
        ));
    }

    fn tiny_nv12_upload<'a>(
        y: &'a [u8],
        uv: &'a [u8],
        y_stride: usize,
        uv_stride: usize,
    ) -> Nv12Upload<'a> {
        Nv12Upload {
            device_id: DeviceId::new(1).expect("device"),
            coded_width: 4,
            coded_height: 4,
            visible_width: 4,
            visible_height: 2,
            y: Nv12Plane {
                width_bytes: 4,
                height: 4,
                stride_bytes: y_stride,
                data: y,
            },
            uv: Nv12Plane {
                width_bytes: 4,
                height: 2,
                stride_bytes: uv_stride,
                data: uv,
            },
            conversion: YcbcrConversion::Rec709Limited,
        }
    }

    #[test]
    fn validates_nv12_geometry_visible_region_and_padding() {
        let y = [
            [16, 64, 128, 235, 0xee, 0xee],
            [32, 80, 144, 220, 0xee, 0xee],
            [48, 96, 160, 210, 0xee, 0xee],
            [64, 112, 176, 200, 0xee, 0xee],
        ]
        .concat();
        let uv = [
            [90, 240, 54, 200, 0xdd, 0xdd],
            [100, 230, 64, 190, 0xdd, 0xdd],
        ]
        .concat();
        let upload = tiny_nv12_upload(&y, &uv, 6, 6);

        let layout = validate_nv12_upload(&upload).expect("valid NV12");

        assert_eq!(layout.y_bytes, 16);
        assert_eq!(layout.uv_bytes, 8);
        assert_eq!(layout.output_samples, 32);
        assert_eq!(layout.cpu_surface_bytes, 36);
        assert_eq!(
            expand_nv12_y_plane(upload.y).expect("Y"),
            vec![16, 64, 128, 235, 32, 80, 144, 220, 48, 96, 160, 210, 64, 112, 176, 200]
        );
        assert_eq!(
            expand_nv12_uv_plane(upload.uv).expect("UV"),
            vec![
                90 | (240 << 8),
                54 | (200 << 8),
                100 | (230 << 8),
                64 | (190 << 8)
            ]
        );
    }

    #[test]
    fn rejects_invalid_nv12_uploads() {
        let y = [0_u8; 16];
        let uv = [0_u8; 8];
        let mut upload = tiny_nv12_upload(&y, &uv, 4, 4);

        upload.coded_width = 3;
        assert!(matches!(
            validate_nv12_upload(&upload),
            Err(ResourceError::InvalidImageDimensions)
        ));

        upload.coded_width = 4;
        upload.y.stride_bytes = 3;
        assert!(matches!(
            validate_nv12_upload(&upload),
            Err(ResourceError::InvalidImageDimensions)
        ));
    }

    #[test]
    fn nv12_reference_uses_420_chroma_addressing() {
        let y = vec![16, 235, 16, 235, 16, 235, 16, 235, 0, 0, 0, 0, 0, 0, 0, 0];
        let uv = vec![128, 128, 128, 128, 200, 16, 16, 200];
        let upload = tiny_nv12_upload(&y, &uv, 4, 4);

        let rgba = nv12_reference_rgba_u16(&upload).expect("reference");

        assert_eq!(&rgba[0..4], &[0, 0, 0, u16::MAX]);
        assert_eq!(&rgba[4..8], &[u16::MAX, u16::MAX, u16::MAX, u16::MAX]);
        assert_eq!(rgba.len(), 4 * 2 * 4);
    }

    #[test]
    fn frame_processor_config_is_bounded() {
        let device_id = DeviceId::new(1).expect("device");
        let valid = GpuFrameProcessorConfig {
            device_id,
            width: 8,
            height: 4,
            slot_count: 3,
            conversion: YcbcrConversion::Rec709Limited,
        };

        assert!(validate_frame_processor_config(valid).is_ok());
        assert!(matches!(
            validate_frame_processor_config(GpuFrameProcessorConfig {
                slot_count: 0,
                ..valid
            }),
            Err(FrameProcessorError::Resource(
                ResourceError::InvalidBufferSize
            ))
        ));
        assert!(matches!(
            validate_frame_processor_config(GpuFrameProcessorConfig {
                slot_count: 9,
                ..valid
            }),
            Err(FrameProcessorError::Resource(
                ResourceError::InvalidBufferSize
            ))
        ));
        assert!(matches!(
            validate_frame_processor_config(GpuFrameProcessorConfig { width: 7, ..valid }),
            Err(FrameProcessorError::Resource(
                ResourceError::InvalidImageDimensions
            ))
        ));
    }

    #[test]
    fn frame_slot_state_exposes_safe_reuse_points() {
        assert!(FrameSlotState::Available.can_submit());
        assert!(!FrameSlotState::Preparing.can_submit());
        assert!(!FrameSlotState::Submitted.can_submit());
        assert!(!FrameSlotState::Completed.can_submit());

        assert!(!FrameSlotState::Available.can_retire());
        assert!(!FrameSlotState::Submitted.can_retire());
        assert!(FrameSlotState::Completed.can_retire());
    }

    #[test]
    fn frame_tokens_are_opaque_and_stable() {
        let token = FrameToken(42);

        assert_eq!(token.get(), 42);
        assert_ne!(token, FrameToken(43));
    }

    #[test]
    fn reference_conversion_uses_422_horizontal_chroma_addressing() {
        let y = [64_u16.to_le_bytes(), 940_u16.to_le_bytes()].concat();
        let cb = [512_u16.to_le_bytes()].concat();
        let cr = [512_u16.to_le_bytes()].concat();
        let upload = tiny_yuv422p10_upload(&y, &cb, &cr, 4, 2);

        let rgba = yuv422p10_reference_rgba_u16(&upload).expect("reference");

        assert_eq!(&rgba[0..4], &[0, 0, 0, u16::MAX]);
        assert_eq!(&rgba[4..8], &[u16::MAX, u16::MAX, u16::MAX, u16::MAX]);
    }

    #[test]
    fn checksum_is_deterministic_for_yuv422p10_output() {
        let values = [1_u16, 2, 3, u16::MAX];

        assert_eq!(
            yuv422p10_rgba_u16_checksum(&values),
            yuv422p10_rgba_u16_checksum(&values)
        );
    }
}
