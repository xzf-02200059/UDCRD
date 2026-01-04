import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import ast

# 设置随机种子以保证结果可复现
np.random.seed(42)
tf.random.set_seed(42)

# 读取CSV文件
df = pd.read_csv('best_dirty_data_episode_6.csv')
print(f"原始数据形状: {df.shape}")

# 准备数据
# 只考虑数值列（不包括weight和fields_to_replace列）
#feature_cols = ['tuple_id', 'src', 'flight', 'sched_dep_time', 'act_dep_time', 'sched_arr_time', 'act_arr_time']
#feature_cols = ['index','ProviderNumber','HospitalName','Address1','Address2','Address3','City','State','ZipCode','CountyName','PhoneNumber','HospitalType','HospitalOwner','EmergencyService','Condition','MeasureCode','MeasureName','Score','Sample','Stateavg']
#feature_cols = ['diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean']
feature_cols = ['index','id','beer_name','style','ounces','abv','ibu','brewery_id','brewery_name','city','state']

# 1. 将所有列转换为数值类型，非数值数据设为NaN
df_numeric = df.copy()
for col in feature_cols:
    #if col in ['src','flight']:
    # if col in ['HospitalName', 'Address1', 'Address2', 'Address3', 'City', 'State', 'CountyName', 'HospitalType',
    #                'HospitalOwner', 'EmergencyService', 'Condition', 'MeasureCode', 'MeasureName','Score' ,'Sample',
    #                'Stateavg']:
    if col in ['beer_name','style','brewery_name','city','state']:
        # 对于字符串列，使用Label Encoding
        df_numeric[col] = pd.factorize(df_numeric[col])[0]
    elif col in ['sched_dep_time', 'act_dep_time', 'sched_arr_time', 'act_arr_time']:
        # 对于时间列，转换为分钟数
        def time_to_minutes(time_str):
            if pd.isna(time_str):
                return np.nan
            try:
                # 处理时间格式，例如 "7:10 a.m."
                time_parts = time_str.split()
                time = time_parts[0]
                period = time_parts[1].lower()
                hours, minutes = map(int, time.split(':'))
                if period == 'p.m.' and hours != 12:
                    hours += 12
                elif period == 'a.m.' and hours == 12:
                    hours = 0
                return hours * 60 + minutes
            except:
                return np.nan
        df_numeric[col] = df_numeric[col].apply(time_to_minutes)
    else:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

# 2. 创建掩码矩阵(mask matrix)来标记错误单元格
error_mask = np.zeros((len(df), len(feature_cols)), dtype=bool)

# 3. 解析fields_to_replace列，标记有错误的单元格
for i, fields_str in enumerate(df['fields_to_replace']):
    if pd.isna(fields_str) or fields_str == '[]':
        continue
    
    try:
        # 尝试不同的解析方法
        if isinstance(fields_str, str):
            if fields_str.startswith('[') and fields_str.endswith(']'):
                # 方法1: 使用ast.literal_eval解析
                try:
                    fields = ast.literal_eval(fields_str)
                except:
                    # 方法2: 手动解析
                    fields_str = fields_str.strip('[]').replace("'", "").replace('"', '')
                    fields = [f.strip() for f in fields_str.split(',')]
            else:
                fields = []
        else:
            fields = []
            
        for field in fields:
            if field in feature_cols:
                col_idx = feature_cols.index(field)
                error_mask[i, col_idx] = True
    except Exception as e:
        print(f"解析错误 第{i}行: {e}")

# 4. 创建好的数据和坏的数据矩阵
# 转换为numpy数组以便处理
data_array = df_numeric[feature_cols].values

# 创建一个掩码数组，标记NaN值
nan_mask = np.isnan(data_array)

# 组合错误掩码和NaN掩码
combined_mask = np.logical_or(error_mask, nan_mask)

# 5. 数据预处理 - 标准化
scaler = StandardScaler()
# 获取非NaN非错误的数据进行标准化拟合
good_mask = ~combined_mask
# 创建一个副本以避免警告
data_for_fit = data_array.copy()
for j in range(data_array.shape[1]):
    col_good_mask = good_mask[:, j]
    if np.any(col_good_mask):
        data_for_fit[col_good_mask, j] = scaler.fit_transform(data_array[col_good_mask, j].reshape(-1, 1)).flatten()

# 为训练VAE准备数据
# 放宽条件：选择至少有一半特征是无错误的行用于训练
min_good_features = len(feature_cols) // 2
good_rows = np.sum(good_mask, axis=1) >= min_good_features
train_data = data_for_fit[good_rows]

# 确保没有NaN值
train_data = np.nan_to_num(train_data)

print(f"用于训练VAE的行数: {train_data.shape[0]}")
print(f"每行平均无错误特征数: {np.mean(np.sum(good_mask, axis=1)):.2f}")

# 6. 构建VAE模型
latent_dim = 4  # 增加潜在空间维度

# 编码器
encoder_inputs = keras.Input(shape=(len(feature_cols),))
x = layers.Dense(128, activation="relu")(encoder_inputs)
x = layers.BatchNormalization()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.BatchNormalization()(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

# 采样层
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Sampling()([z_mean, z_log_var])

# 解码器
decoder_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(32, activation="relu")(decoder_inputs)
x = layers.BatchNormalization()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
decoder_outputs = layers.Dense(len(feature_cols))(x)

# 定义编码器与解码器模型
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

# VAE模型
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # 修改损失函数计算，增加数值稳定性
            reconstruction_loss = tf.reduce_mean(
                tf.square(data - reconstruction)
            )
            
            # 添加KL损失权重
            kl_weight = 0.01
            kl_loss = -0.5 * kl_weight * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            
            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        # 梯度裁剪
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# 使用较小的学习率
optimizer = keras.optimizers.Adam(learning_rate=0.001)
vae = VAE(encoder, decoder)
vae.compile(optimizer=optimizer)

# 7. 训练VAE模型
print("开始训练VAE模型...")
# 添加早停机制
early_stopping = keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=5,
    restore_best_weights=True
)
vae.fit(train_data, epochs=50, batch_size=32, callbacks=[early_stopping], verbose=1)
print("VAE模型训练完成！")

# 8. 重构所有数据
# 为所有行（包括错误行）生成重构值
all_data_no_nan = np.nan_to_num(data_for_fit)
z_mean, _, _ = vae.encoder.predict(all_data_no_nan)
reconstructed_data = vae.decoder.predict(z_mean)

# 9. 计算重构误差（仅针对错误单元格）
reconstruction_errors = np.zeros(len(df))

for i in range(len(df)):
    if df['weight'][i] == 1:  # 只对有错误的行计算重构误差
        # 计算错误单元格的重构误差
        error_indices = np.where(combined_mask[i])[0]
        if len(error_indices) > 0:
            original = data_array[i, error_indices]
            reconstructed = reconstructed_data[i, error_indices]
            
            # 过滤掉原始数据中的NaN值
            valid_indices = ~np.isnan(original)
            if np.any(valid_indices):
                error = np.mean(np.square(original[valid_indices] - reconstructed[valid_indices]))
                reconstruction_errors[i] = error

# 10. 替换错误单元格为-999999.0
result_df = df.copy()
for i in range(len(df)):
    for j, col in enumerate(feature_cols):
        if combined_mask[i, j]:
            result_df.loc[i, col] = -999999.0

# 11. 添加重构误差列
result_df['reconstruction_error'] = reconstruction_errors
result_df = result_df.drop(columns=['weight', 'fields_to_replace'])
# 12. 保存结果
output_file = 'reconstructed_data.csv'
result_df.to_csv(output_file, index=False)
print(f"处理完成！结果已保存到 {output_file}")

# 13. 输出一些汇总统计信息
print("\n===== 汇总统计 =====")
print(f"总行数: {len(df)}")
print(f"有错误的行数 (weight=1): {sum(df['weight'] == 1)}")
print(f"被标记为错误的单元格总数: {np.sum(error_mask)}")
print(f"NaN值的总数: {np.sum(nan_mask)}")
print(f"被替换为-999999.0的单元格总数: {np.sum(combined_mask)}")
print(f"平均重构误差: {np.mean(reconstruction_errors[reconstruction_errors > 0])}") 