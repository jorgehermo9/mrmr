
use std::error::Error;
use std::io;
use std::collections::HashMap;


struct Dataset{
	class:String,
	features:Vec<String>,
	features_values: HashMap<String,Vec<String>>,
	datasize:usize,
	features_probs:HashMap<String,HashMap<String,f64>>
}
impl Dataset {
	fn new(class:String, features:Vec<String>, features_values:HashMap<String,Vec<String>>,datasize:usize) -> Dataset{
		let mut dataset =  Dataset{
			class:String::from(class),
			features,
			features_values,
			datasize,
			features_probs:HashMap::new(),
		};
		dataset.get_single_probs();	
		dataset
	}
	fn get_single_probs(& mut self){
		let mut target_features = self.features.clone();
		target_features.push(String::from(&self.class));
		for feature in &target_features{
			let feature_data = get_feature_data(self.features_values.get(feature).unwrap());
			let feature_probs = get_probs(&feature_data,self.datasize);
			self.features_probs.insert(String::from(feature),feature_probs);
		}
	}
}
struct MrmrInfo<'a>{
	relevance_map:HashMap<&'a String,f64>,
	accum_redundancy:HashMap<String,f64>,
	selected_features:Vec<(String,f64)>,
	remaining_features:Vec<String>,
}
fn mrmr(dataset_info:&Dataset){
	let mut mrmr_info = MrmrInfo{
		relevance_map: HashMap::new(),
		accum_redundancy:HashMap::new(),
		selected_features: Vec::new(),
		remaining_features: dataset_info.features.clone(),
	};

	mrmr_info.relevance_map = get_relevance_vector(dataset_info);
	let (max_relevance_feature,max_relevance) = get_max_value(&mrmr_info.relevance_map);
	
	mrmr_info.selected_features.push((String::from(max_relevance_feature),max_relevance));
	mrmr_info.remaining_features.retain(|feature| feature != max_relevance_feature);
	
	let mut last_feature = String::from(max_relevance_feature);
	println!("selected {}",last_feature);
	
	for _ in 0.. mrmr_info.remaining_features.len(){
		let mut m_info_map:HashMap<&String,f64> = HashMap::new();
		for target_feature in &mrmr_info.remaining_features{
			let redundancy = mutual_info(dataset_info,(&last_feature,target_feature));
			let accum_redundancy = mrmr_info.accum_redundancy.entry(String::from(target_feature)).or_insert(0.0);
			*accum_redundancy+= redundancy;
			//Accum redundancy = sum of redundancy between target and currently selected features
			let relevance = *mrmr_info.relevance_map.get(target_feature).unwrap();

			m_info_map.insert(target_feature,get_mrmr(relevance, *accum_redundancy/dataset_info.datasize as f64));
		}
		let (max_mrmr_feature,max_mrmr) = get_max_value(&m_info_map);
		last_feature = String::from(max_mrmr_feature);

		mrmr_info.selected_features.push((String::from(&last_feature),max_mrmr));
		mrmr_info.remaining_features.retain(|feature| feature != &last_feature);
		
		println!("selected {}",last_feature);
	}
	for (index,(feature,value)) in mrmr_info.selected_features.iter().enumerate() {
		println!("{}. {} -> {}",index,feature,value);
	}
	

}
fn get_features_values(entities: &Vec<csv::StringRecord>, features: &Vec<String>)
	-> HashMap<String,Vec<String>>{
	let mut feature_values: HashMap<String,Vec<String>>= HashMap::new();
	for feature in features{
		feature_values.insert(String::from(feature),Vec::new());
	}

	//Set feature value for each entity and feature on the dataset
	for result in entities {
		for (index,feature) in features.iter().enumerate() {
			feature_values.get_mut(feature).unwrap().push(String::from(result.get(index).unwrap()));
		}
	}
	feature_values
}
fn get_feature_data(target_feature: &Vec<String>) -> HashMap<String,u32>{
	let mut map: HashMap<String,u32> = HashMap::new();

	for target_value in target_feature{
		if let Some(value)= map.get_mut(target_value){
			*value+=1;
		}else{
			map.insert(String::from(target_value), 1);
		}
	}
	map
}
fn get_probs<Q>(feature_data: &HashMap<Q,u32>,datasize:usize) -> HashMap<Q,f64>
	where Q:Clone+Eq + std::hash::Hash{
	
	let mut probs: HashMap<Q,f64> = HashMap::new();
	for (key,value) in feature_data{
		probs.insert(key.clone(),*value as f64 / datasize as f64);
	}
	probs

}
fn intersection(dataset_info:&Dataset,features:(&str,&str))
	->HashMap<(String,String),u32>{
	
	let mut intersect: HashMap<(String,String),u32> = HashMap::new();

	let (a,b) = features;

	let a_values = dataset_info.features_values.get(&String::from(a)).unwrap();
	let b_values = dataset_info.features_values.get(&String::from(b)).unwrap();
	
	let datasize = a_values.len();
	for i in 0..datasize{
		let a_value = a_values.get(i).unwrap();
		let b_value = b_values.get(i).unwrap();
		
		let pair = (String::from(a_value), String::from(b_value));
		
		if let Some(value)= intersect.get_mut(&pair){
			*value+=1;
		}else{
			intersect.insert(pair, 1);
		}
	}
	intersect
}

fn mutual_info(dataset_info:&Dataset,features_pair:(&str,&str)) -> f64{

	let (a_feature,b_feature) = features_pair;
	let a_probs = dataset_info.features_probs.get(a_feature).unwrap();
	let b_probs = dataset_info.features_probs.get(b_feature).unwrap();

	let intersect_probs = get_probs(&intersection(dataset_info,(a_feature,b_feature)),dataset_info.datasize);

	let mut m_info:f64=0.0;
	for (a_instance,_) in a_probs.iter(){
		for (b_instance,_) in b_probs.iter(){
			if let Some(a_and_b) = intersect_probs.get(&(a_instance.clone(),b_instance.clone())){
				let a = a_probs.get(a_instance).unwrap() as &f64;
				let b = b_probs.get(b_instance).unwrap() as &f64;

				let mut calc= a_and_b as &f64/(a * b );
				calc = calc.log10() * a_and_b as &f64;
				m_info += calc;
			}//Else: m_info+=0
			
		}
	}
	m_info
}
fn get_relevance_vector<'a>(dataset_info: &'a Dataset)
	->  HashMap<&String,f64>{
	
	let mut relevance_map: HashMap<&String,f64> = HashMap::new();

	for feature in &dataset_info.features{
		let m_info = mutual_info(dataset_info,(feature,"class"));
		relevance_map.insert(feature,m_info);
	}
	relevance_map
}

fn get_mrmr(relevance:f64,redundancy:f64) ->f64{
	relevance-redundancy
}
fn get_max_value<'a>(data: &'a HashMap<&String,f64>)-> (&'a String,f64) {
	let mut max_feature= data.iter().next().unwrap().0;
	let mut max_value:f64 =*data.get(max_feature).unwrap();
	for (index,(feature,value)) in data.iter().enumerate() {
		if *value>max_value || index == 0{
			max_feature = feature;
			max_value = *value;
		} 
	}
	(max_feature,max_value)
}

fn read_csv() -> Result<(),Box<dyn Error>>{

	let mut rdr = csv::Reader::from_reader(io::stdin());
	let mut features:Vec<String> = Vec::new();
	for feature in rdr.headers()?{
		features.push(String::from(feature))
	}
	let mut entities: Vec<csv::StringRecord> = Vec::new();
	//Save each entry of the dataset
	for result in rdr.records(){
		let record = result?;
		entities.push(record);
	}
	let datasize = entities.len();
	println!("Read {} lines",datasize);
	println!("features: {:?}", features);

	
	let features_values = get_features_values(&entities,&features);

	let class = "class";
	features.retain(|feature| feature != "Weight" && feature != "Height" && feature !=class);
	//features.retain(|feature|  feature !=class);
	
	let dataset_info = Dataset::new(String::from(class),features,features_values,datasize);
	mrmr(&dataset_info);

	
	Ok(())
}

fn main() {

	if let Err(err) = read_csv(){
		panic!("Error reading csv: {}",err)
	}
}
