
use std::error::Error;
use std::fs;
use std::io::{self,BufRead,BufReader};
use std::collections::HashMap;
use clap::Parser;

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
	dataset_info: &'a Dataset,
	num_features:usize,
	relevance_map:HashMap<&'a String,f64>,
	accum_redundancy:HashMap<String,f64>,
	selected_features:Vec<(String,f64)>,
	remaining_features:Vec<String>,
}
impl MrmrInfo<'_>{
	fn new(dataset_info:&Dataset,num_features:usize) -> MrmrInfo{
		MrmrInfo{
			dataset_info,
			num_features,
			relevance_map:HashMap::new(),
			accum_redundancy:HashMap::new(),
			selected_features:Vec::new(),
			remaining_features:dataset_info.features.clone(),
		}
	}
	fn select_features(&mut self)-> &Vec<(String,f64)> {
		self.relevance_map = get_relevance_values(self.dataset_info);
		let (max_relevance_feature,max_relevance) = get_max_value(&self.relevance_map);
		
		self.selected_features.push((String::from(max_relevance_feature),max_relevance));
		self.remaining_features.retain(|feature| feature != max_relevance_feature);
		
		let mut last_feature = String::from(max_relevance_feature);
		
		// -1 features, already 1 selected
		for _ in 0..self.num_features-1{

			let mut m_info_map:HashMap<&String,f64> = HashMap::new();
			for target_feature in &self.remaining_features{
				let redundancy = mutual_info(self.dataset_info,(&last_feature,target_feature));
				let accum_redundancy = self.accum_redundancy.entry(String::from(target_feature)).or_insert(0.0);
				*accum_redundancy+= redundancy;
				//Accum redundancy = sum of redundancy between target and currently selected features
				let relevance = *self.relevance_map.get(target_feature).unwrap();
	
				m_info_map.insert(target_feature,get_mrmr(relevance, *accum_redundancy/self.selected_features.len() as f64));
			}
			let (max_mrmr_feature,max_mrmr) = get_max_value(&m_info_map);
			last_feature = String::from(max_mrmr_feature);
	
			self.selected_features.push((String::from(&last_feature),max_mrmr));
			self.remaining_features.retain(|feature| feature != &last_feature);	
			println!("Selected {}",&last_feature);		
		}
		&self.selected_features
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
		let value = map.entry(String::from(target_value)).or_insert(0);
		*value+=1;
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
fn intersection<'a>(dataset_info:&'a Dataset,features:(&str,&str))
	->HashMap<(&'a String,&'a String),u32>{
	
	let mut intersect: HashMap<(&String,&String),u32> = HashMap::new();

	let (a,b) = features;

	let a_values = dataset_info.features_values.get(a).unwrap();
	let b_values = dataset_info.features_values.get(b).unwrap();
	
	let datasize = a_values.len();
	for i in 0..datasize{
		let a_value = a_values.get(i).unwrap();
		let b_value = b_values.get(i).unwrap();
		
		let pair = (a_value, b_value);
		
		let value = intersect.entry(pair).or_insert(0);
		*value+=1;
	}
	intersect
}

fn mutual_info(dataset_info:&Dataset,features_pair:(&str,&str)) -> f64{

	let (a_feature,b_feature) = features_pair;
	let a_probs = dataset_info.features_probs.get(a_feature).unwrap();
	let b_probs = dataset_info.features_probs.get(b_feature).unwrap();

	let intersect_probs = get_probs(&intersection(dataset_info,(a_feature,b_feature)),dataset_info.datasize);

	let mut m_info:f64=0.0;

	for ((a_instance,b_instance),a_and_b_prob) in intersect_probs{
		let a_instance_prob = a_probs.get(a_instance).unwrap();
		let b_instance_prob = b_probs.get(b_instance).unwrap();

		//H. Peng utilizes log2 in his implementation instead of log10.
		let calc = a_and_b_prob * (a_and_b_prob/(a_instance_prob*b_instance_prob)).log2();
		m_info+=calc;
	}
	m_info
}
fn get_relevance_values<'a>(dataset_info: &'a Dataset)
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

fn read_csv(cli:Args) -> Result<(),Box<dyn Error>>{

	let reader:Box<dyn BufRead>= match cli.csv{
		None => Box::new(BufReader::new(io::stdin())),
		Some(csv) => Box::new(BufReader::new(fs::File::open(csv).unwrap()))
	};
	
	let mut csv_reader = csv::Reader::from_reader(reader);

	let mut features:Vec<String> = Vec::new();
	for feature in csv_reader.headers()?{
		features.push(String::from(feature))
	}
	let mut entities: Vec<csv::StringRecord> = Vec::new();
	//Save each entry of the dataset
	for result in csv_reader.records(){
		let record = result.unwrap();
		entities.push(record);
	}
	let datasize = entities.len();
	println!("Read {} lines",datasize);
	println!("features: {:?}", features);

	
	let features_values = get_features_values(&entities,&features);

	let class = cli.class;
	features.retain(|feature| feature !=&class);
	
	let dataset_info = Dataset::new(String::from(class),features,features_values,datasize);

	if let Some(features) = cli.num_features{
		assert!(features > 0)
	}
	let num_features = match cli.num_features{
		Some(features) => if features < dataset_info.features.len() {features} else {dataset_info.features.len()},
		None => dataset_info.features.len()
	};
	let mut mrmr_info = MrmrInfo::new(&dataset_info,num_features);
	mrmr_info.select_features();

	for (index,(feature,value)) in mrmr_info.selected_features.iter().enumerate() {
		println!("{}. {} -> {}",index+1,feature,value);
	}
	
	Ok(())
}

#[derive(Parser,Debug)]
#[clap(author,version,about)]
struct Args{
	
	//Csv file to read (if none specified, it reads from stdin)
	#[clap(short,long)]
	csv:Option<String>,
	
	//Class feature
	#[clap(long,default_value_t = String::from("class"))]
	class:String,

	//Number of features to be selected
	#[clap(short,long)]
	num_features:Option<usize>,
}
fn main() {
	let cli = Args::parse();
	println!("{:?}", cli);
	if let Err(err) = read_csv(cli){
		panic!("Error reading csv: {}",err)
	}
}
