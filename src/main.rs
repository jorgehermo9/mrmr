
use std::error::Error;
use std::io;
use std::collections::HashMap;


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
fn intersection(features_values:&HashMap<String,Vec<String>>,features:(&str,&str))
	->HashMap<(String,String),u32>{
	
	let mut intersect: HashMap<(String,String),u32> = HashMap::new();

	let (a,b) = features;

	let a_values = features_values.get(&String::from(a)).unwrap();
	let b_values = features_values.get(&String::from(b)).unwrap();
	
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

fn mutual_info(features_values:&HashMap<String,Vec<String>>,datasize:usize,features_pair:(&str,&str),data_pair:(&HashMap<String,u32>,&HashMap<String,u32>)) -> f64{

	let (a,b) = data_pair;
	let (a_feature,b_feature) = features_pair;
	let a_probs = get_probs(a,datasize);
	let b_probs = get_probs(b,datasize);
	let intersect_probs = get_probs(&intersection(features_values,(a_feature,b_feature)),datasize);

	let mut m_info:f64=0.0;
	for (a_instance,_) in a.iter(){
		for (b_instance,_) in b.iter(){
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
fn get_relevance_vector<'a>(features_values:&HashMap<String,Vec<String>>,datasize:usize,class:&str,features:&'a Vec<String>)
	->  Vec<(&'a String,f64)>{
	
	let class_data = get_feature_data(features_values.get(class).unwrap());
	let mut mutual_vector: Vec<(&String,f64)> = Vec::new();

	for feature in features{
		if feature.eq(class){
			continue;
		}
		let feature_data = get_feature_data(features_values.get(feature).unwrap());
		let m_info = mutual_info(&features_values,datasize,(feature,"class"),(&feature_data,&class_data));
			mutual_vector.push((feature,m_info));
	}
	mutual_vector
}
fn get_redundancy(features_values:&HashMap<String,Vec<String>>,datasize:usize,target_feature:&str,class:&str,features:&[String])
	-> f64{
	let mut redundancy:f64 = 0.0;
	let target_data = get_feature_data(features_values.get(target_feature).unwrap());
	for feature in features{
		if feature.eq(class)|| feature.eq(target_feature){
			continue;
		}
		let feature_data = get_feature_data(features_values.get(feature).unwrap());
		let m_info = mutual_info(features_values,datasize,(target_feature,feature),(&target_data,&feature_data));
		redundancy+=m_info;
	}
	redundancy/features.len() as f64
}
fn get_mrmr(relevance:f64,redundancy:f64) ->f64{
	relevance-redundancy
}
fn get_max_value<'a>(vec: &'a Vec<(&String,f64)>)-> (&'a String,f64) {
	let (mut max_feature,mut max_val) = vec.get(0).unwrap();
	for (feature,value) in vec{
		if *value > max_val{
			max_val = *value;
			max_feature = feature;
		}
	}
	(max_feature,max_val)
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

	features.retain(|feature| feature != "Weight" && feature != "Height");

	let relevance_vector = get_relevance_vector(&features_values,datasize,"class",&features);
	for pair in &relevance_vector{
		let (feature,value) = pair;
		println!("Relevance: {} -> {}",feature,value);
	}


	let (max_feature,max_relevance) = get_max_value(&relevance_vector);
	println!("Max relevance: {} ->{}",max_feature,max_relevance);

	let mut remaining_features = features.clone();
	remaining_features.retain(|feature| feature != "class");

	let mut selected_features: Vec<(String,f64)>= Vec::new();
	selected_features.push((String::from(max_feature),max_relevance));
	remaining_features.retain(|feature| feature != max_feature);
	let mut last_feature= String::from(max_feature);

	// Implementación comparando sólo con la ultima seleccionada
	for _ in 0..remaining_features.len(){
		let mut current_mrmr_vector: Vec<(&String,f64)> = Vec::new();
		for target_feature in &remaining_features {
			//iterate only over remaining features
			if !remaining_features.contains(target_feature) {
				continue;
			}
			let mut relevance:f64 =0.0;
			for (feature,value) in &relevance_vector{
				if target_feature.eq(*feature){
					relevance = *value;
				}
			}
			let last_feature_vec = vec![String::from(&last_feature)];
			//Get redundancy comparing to selected values
			let red = get_redundancy(&features_values, datasize,target_feature, "class", &last_feature_vec);
			current_mrmr_vector.push((target_feature,get_mrmr(relevance,red)));
		}
		let (max_mrmr_feature,max_mrmr) = get_max_value(&current_mrmr_vector);
		last_feature = String::from(max_mrmr_feature);

		selected_features.push((String::from(max_mrmr_feature),max_mrmr));
		remaining_features.retain(|feature| feature != &last_feature);

		println!("selected {}",last_feature)
	}
	
	for (index,(feature,value)) in selected_features.iter().enumerate() {
		println!("{}. {} -> {}",index+1,feature,value);
	}

	/* Implementación comparando todos con todos
	let mut current_mrmr_vector: Vec<(&String,f64)> = Vec::new();
	for target_feature in &features{
		if target_feature == "class" || target_feature == max_feature{ 
			continue;
		}
		let mut relevance:f64 =0.0;
		for (feature,value) in &relevance_vector{
			if target_feature.eq(*feature){
				relevance = *value;
			}
		}
		//Get redundancy comparing to selected values
		let red = get_redundancy(&features_values, datasize,target_feature, "class", &features);
		current_mrmr_vector.push((target_feature,get_mrmr(relevance,red)));
		println!("Calculated for {}",target_feature);
	}

	for _ in 0..current_mrmr_vector.len() {
		
		let (mut max_feature,mut max) = current_mrmr_vector[0];
		for (feature,value) in &current_mrmr_vector{
			if *value>max{
				max = *value;
				max_feature = feature;
			}
		}
		current_mrmr_vector.retain(|(feature,_)| *feature != max_feature);
		selected_features.push((String::from(max_feature),max));
	}
	
	for (index,(feature,value)) in selected_features.iter().enumerate() {
		println!("{}. {} -> {}",index+1,feature,value);
	}*/

	
	


	Ok(())
}

fn main() {

	if let Err(err) = read_csv(){
		panic!("Error reading csv: {}",err)
	}
}
