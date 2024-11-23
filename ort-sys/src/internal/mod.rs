use std::hash::{BuildHasher, Hasher, RandomState};

pub mod dirs;
<<<<<<< HEAD

#[cfg(feature = "download-binaries")]
include!(concat!(env!("OUT_DIR"), "/downloaded_version.rs"));
||||||| parent of eb51646 (fix: concurrent downloads, ref #322)
=======

pub fn random_identifier() -> String {
	let mut state = RandomState::new().build_hasher().finish();
	std::iter::repeat_with(move || {
		state ^= state << 13;
		state ^= state >> 7;
		state ^= state << 17;
		state
	})
	.take(12)
	.map(|i| b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"[i as usize % 62] as char)
	.collect()
}
>>>>>>> eb51646 (fix: concurrent downloads, ref #322)
